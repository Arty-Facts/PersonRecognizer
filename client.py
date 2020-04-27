import io
import socket
import struct
from PIL import Image
import matplotlib.pyplot as ply
from hidden import IP

client_socket = socket.socket()

client_socket.connect((IP, 8000)) 

# Make a file-like object out of the connection
connection = client_socket.makefile('rb')
try:
    img = None
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image.open(image_stream)
        
        if img is None:
            img = pl.imshow(image)
        else:
            img.set_data(image)

        ply.pause(0.01)
        ply.draw()

        print('Image is %dx%d' % image.size)
        image.verify()
        print('Image is verified')
finally:
    connection.close()
    server_socket.close()
