## Description
A Multilingual Covid-19 dashboard built using plotly Dash, used Hugging face transformer - MarianMT for translation.
Also has live news feed built using newsapi.
Single language version of this app is currently deployed in heroku - [here](https://dash-covid19-multilingual.herokuapp.com/)

### Screenshots - 

![screenshot1](https://github.com/nagarajbhat/dash-covid19-multilingual/blob/master/screenshots/screenshot1.PNG)
![screenshot2](https://github.com/nagarajbhat/dash-covid19-multilingual/blob/master/screenshots/screenshot2.PNG)
![screenshot3](https://github.com/nagarajbhat/dash-covid19-multilingual/blob/master/screenshots/screenshot3.PNG)
![screenshot4](https://github.com/nagarajbhat/dash-covid19-multilingual/blob/master/screenshots/screenshot4.PNG)
![screenshot5](https://github.com/nagarajbhat/dash-covid19-multilingual/blob/master/screenshots/screenshot6.PNG)


## Instrucitons to run locally.
1. Clone this repo
```
git clone https://github.com/nagarajbhat/dash-covid19-multilingual.git
```

2. Create and activate virtual environment (windows)
```
python -m virtualenv venv
cd venv/Scripts
activate
```

3. Come back to project folder and install dependencies using requirements.txt
```
pip install -r requirements.txt
```

4. Go to [newsapi](https://newsapi.org/docs) , create an account and copy your api key, and replace the api key in app.py 
```
newsapi = NewsApiClient(api_key='<YOUR_API_KEY>')
```

5. execute the app from the command line
```
python app.py
```

6. open in browser
the app will be served at https://localhost:8050
