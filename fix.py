with open('app.py', 'r') as f:
    content = f.read()

old = '''    port = int(os.environ.get("PORT", 7860))
app.run(debug=False, host="0.0.0.0", port=port)'''

new = '''    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)'''

content = content.replace(old, new)
with open('app.py', 'w') as f:
    f.write(content)
print("Fixed!")
