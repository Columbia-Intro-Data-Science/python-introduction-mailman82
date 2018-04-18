import webtest

import main


def test_get():
    app = webtest.TestApp(main.app)

    response = app.get('/')

    assert response.status_int == 200
    assert response.body == 'Hello, Liam, it works now get to work!'
