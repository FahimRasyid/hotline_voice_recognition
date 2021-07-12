from flask import Flask, render_template
app = Flask(__name__, static_url_path='')


@app.route('/')
def index(name="Fahim Rasyid"):
    return render_template('index.html', name=name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True ,port=5000)