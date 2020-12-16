from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
import model
import json

print('init model')
model = model.Model()

@view_config(
    route_name='get_data',
    renderer='json',
    request_method='GET'
)
def get_data(request):
    print(request.body)
    data = json.loads(request.body)
    print('text:  ', data['text'])
    out = model.predict(str(data['text']))
    return {"output": out}

if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('get_data', '/get_data')
        config.scan()
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()