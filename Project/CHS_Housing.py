import webapp2


class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Hello, Liam - This works, now you get to work!')


app = webapp2.WSGIApplication([
    ('/', MainPage),
], debug=True)
