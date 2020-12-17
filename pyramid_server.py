from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.events import NewRequest
from pyramid.view import view_config
import model

# add additional header for allowing cross origin requests and allow methods
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
    route_name='predict_sentiment',
    renderer='json',
    request_method='POST'
)
def predict_sentiment(request):
    # read data from request
    body = request.swagger_data["body"]
    text = body['text']
    print("text : ", text)
    # call model and predict from text
    out = model.predict(str(text))
    return {"output": out}

if __name__ == '__main__':
    # setting up swagger
    settings=dict()
    settings['pyramid_swagger.schema_directory'] = 'api-docs/'
    settings['pyramid_swagger.enable_swagger_spec_validation'] = True
    settings['pyramid_swagger.enable_request_validation'] = True
    settings['pyramid_swagger.use_models'] = True
    with Configurator(settings=settings) as config:
        config.include('pyramid_swagger')
        config.add_static_view('static', 'static', cache_max_age=3600)
        # allow cross origin requests
        config.add_subscriber(add_cors_headers_response_callback, NewRequest)
        # add request route
        config.add_route('predict_sentiment', '/predict_sentiment')
        config.scan()
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    print("\n\n\nReady for serving  \n\n")
    server.serve_forever()