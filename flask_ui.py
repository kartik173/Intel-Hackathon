from flask import Flask,render_template


app = Flask(__name__)

@app.route('/')
def index():
	f=open('data.txt','r')
	li=f.readlines()
	li1=[]
	for i in li:
		if   len(i.strip()) > 0:
			i=i.strip()
			li1.append(i.split(","))
	return render_template('new.html',res=li1)
	


if __name__ == '__main__':
	app.run(debug=True)
