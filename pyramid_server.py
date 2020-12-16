from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.events import NewRequest
from pyramid.view import view_config
import model
import json

def add_cors_headers_response_callback(event):
    def cors_headers(request, response):
        response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST,GET,DELETE,PUT,OPTIONS',
        'Access-Control-Allow-Headers': 'Origin, Content-Type, Accept, Authorization',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '1728000',
        })
    event.request.add_response_callback(cors_headers)


print('init model')
model = model.Model()

@view_config(
    route_name='get_data',
    renderer='json',
    request_method='POST'
)
def get_data(request):
    print(request)
    data = json.loads(request.body)
    print('text:  ', data['text'])
    out = model.predict(str(data['text']))
    return {"output": out}

if __name__ == '__main__':
    with Configurator() as config:
        config.add_subscriber(add_cors_headers_response_callback, NewRequest)
        config.add_route('get_data', '/get_data')
        config.scan()
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()