var http = require('http');

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'application/json'});
  console.log("Json req", req.body)
  
  res.end(req.body);
}).listen(8080);

