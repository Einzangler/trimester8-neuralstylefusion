import os, uuid, threading, time
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Job store: {job_id: {status, progress, message, result}}
jobs = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- NST helpers (from notebook) ----------

def image_to_tensor(pil_img, size=300):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0)

def tensor_to_pil(tensor):
    img = tensor.to("cpu").clone().detach().squeeze()
    img = img.permute(1, 2, 0).numpy().clip(0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def get_features(image, model):
    layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}
    features = {}
    x = image
    for name, layer in enumerate(model):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def gram_matrix(tensor):
    _, n_c, n_h, n_w = tensor.size()
    tensor = tensor.view(n_c, n_h * n_w)
    return torch.mm(tensor, tensor.t())

def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE)
    for p in vgg.parameters():
        p.requires_grad_(False)
    vgg.eval()
    return vgg

# ---------- Background worker ----------

def run_nst(job_id, content_path, style_path, alpha, beta, iterations, img_size):
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = 'Loading VGG-19...'
        jobs[job_id]['progress'] = 2

        vgg = load_vgg()

        content_img = Image.open(content_path)
        style_img   = Image.open(style_path)

        content = image_to_tensor(content_img, img_size).to(DEVICE)
        style   = image_to_tensor(style_img,   img_size).to(DEVICE)

        jobs[job_id]['message'] = 'Extracting features...'
        jobs[job_id]['progress'] = 5

        content_features = get_features(content, vgg)
        style_features   = get_features(style,   vgg)
        gm = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        G = content.clone().requires_grad_(True).to(DEVICE)

        style_weights = {'conv1_1':0.85,'conv2_1':0.56,'conv3_1':0.11,'conv4_1':0.15,'conv5_1':0.2}
        optimizer = torch.optim.Adam([G], lr=0.02)
        loss_fn   = torch.nn.MSELoss()

        jobs[job_id]['message'] = 'Stylizing image...'

        for i in range(iterations):
            target_features = get_features(G, vgg)
            content_loss = loss_fn(target_features['conv4_2'], content_features['conv4_2'])

            style_loss = 0
            for layer in style_weights:
                tg = gram_matrix(target_features[layer])
                style_loss += style_weights[layer] * loss_fn(tg, gm[layer])

            total_loss = alpha * content_loss + beta * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pct = 5 + int((i + 1) / iterations * 90)
            jobs[job_id]['progress'] = pct
            jobs[job_id]['message']  = f'Iteration {i+1}/{iterations} — loss: {total_loss.item():.1f}'

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.jpg")
        tensor_to_pil(G).save(out_path, quality=92)

        jobs[job_id].update({'status':'done','progress':100,'message':'Complete!','result': out_path})

    except Exception as e:
        jobs[job_id].update({'status':'error','message': str(e)})

# ---------- Routes ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stylize', methods=['POST'])
def stylize():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Both content and style images required'}), 400

    alpha      = float(request.form.get('alpha', 100))
    beta       = float(request.form.get('beta', 40))
    iterations = int(request.form.get('iterations', 500))
    img_size   = int(request.form.get('img_size', 256))

    iterations = min(max(iterations, 50), 1000)
    img_size   = min(max(img_size, 128), 512)

    job_id = str(uuid.uuid4())[:8]
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    c_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_content.jpg")
    s_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_style.jpg")
    Image.open(request.files['content']).convert("RGB").save(c_path)
    Image.open(request.files['style']).convert("RGB").save(s_path)

    jobs[job_id] = {'status':'queued','progress':0,'message':'Queued...','result':None}
    t = threading.Thread(target=run_nst, args=(job_id, c_path, s_path, alpha, beta, iterations, img_size), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})

@app.route('/api/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error':'Not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/api/result/<job_id>')
def result(job_id):
    if job_id not in jobs or jobs[job_id]['status'] != 'done':
        return jsonify({'error':'Not ready'}), 404
    return send_file(jobs[job_id]['result'], mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
