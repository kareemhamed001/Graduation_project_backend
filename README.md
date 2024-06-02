<h1 ><span style="color: #22cce5">Scene</span> Scanner</h1>
<p>Scene Scanner is an AI-based tool to scan a scene and detect if it's fake or real. It utilizes the Hair cascade model to detect human faces in the scene and employs a custom DNN trained to discern real and fake faces.</p>
<p>The tool is hosted on a landing page built using Flask, websockets, and multithreading to handle multiple clients simultaneously in real-time.</p>
<p>The tool is developed using Python and OpenCV.</p>
<h2>Used technologies</h2>
<ul>
<li>Python</li>
<li>OpenCV</li>
<li>Flask</li>
<li>Websockets</li>
<li>HTML</li>
<li>CSS</li>
<li>JavaScript</li>
</ul>
<h2>How to run</h2>
<ol>
<li>Clone the repository</li>
<li>Install the required packages using <code>pip install -r requirements.txt</code></li>
<li>Build Css <code>tailwindcss -i ./static/css/input.css -o ./static/css/output.css --watch
</code></li>
<li>Run the Flask app using <code>flask --app main --debug run</code></li>
<li>Open the browser and go to <code>http://127.0.0.1:5000/</code></li>
</ol>
<h2>How it works</h2>
<p>The tool uses the Hair cascade model to detect human faces in the scene. It then crops the detected faces and passes them to a custom DNN model trained to discern real and fake faces. The model is trained using a dataset of real and fake faces.</p>
<p>The tool displays real time results on the same page using websockets.</p>
<p><span style="font-weight: bold">Note</span> : make sure your computer connected to the internet to load the cdn</p>
<h2>Contributors</h2>
<ul>
<li><a href="https://github.com/kareemhamed001">@kareemhamed001</a></li>
<li><a href="">@ahmed_anwar</a></li>
<li><a href="">@alaa_taha</a></li>
<li><a href="">@ashrakat_anwer</a></li>
<li><a href="">@romaisaa_khaled</a></li>
<li><a href="">@sondos_mohamed</a></li>
</ul>
<h2>License</h2>
<p>MIT</p>
<p>Copyright (c) 2024 Scene Scanner</p>
<p>This project is developed by six students from the Faculty of Computers and Artificial Intelligence, Suez University, as a graduation project.</p>


