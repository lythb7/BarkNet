import Queue
import time
import threading

import sounddevice as sd
import serial
import pynmea2
import tensorflow as tf

import model
import util
import wiringpi

print("Initializing CNN...")
myModel = model.BarkCNN()


gps_coords = ()
q = Queue.Queue()
t_now1 = time.time()

valid = False
gps_mutex = threading.Lock()
text_mutex = threading.Lock()

wiringpi.wiringPiSetup();
wiringpi.pinMode(1, wiringpi.OUTPUT)

ser = serial.Serial("/dev/ttyAMA0", 115200)

sd.default.channels = 1
sd.default.samplerate = 44100
sd.default.device = 'hw:1,0'

counter = 0

print("Acquiring GPS Location...")
ser.write("AT+GPS=1\r\n")
time.sleep(0.05)
ser.reset_input_buffer()
ser.write("AT+GPSRD=1\r\n")
time.sleep(0.05)
ser.reset_input_buffer()
time.sleep(1)
ser.reset_input_buffer()

def gps_fix():
	global valid
	global gps_coords
	while True:
		time.sleep(0.01)
		text_mutex.acquire()
		#print("Got the text lock")
		try:
			data = ser.read_until("\r\n")
			try:
				fix = pynmea2.parse(data[7:])
				if fix.is_valid:
					gps_mutex.acquire()
					#print("Got the gps lock")
					try:
						gps_coords = (fix.lat, fix.lon)
						valid = True
					finally:
						gps_mutex.release()
				else:
					gps_mutex.acquire()
					#print("Got the gps lock")
					try:
						valid = False
					finally:
						gps_mutex.release()
			
			except pynmea2.nmea.ParseError:
				pass
		finally:
			text_mutex.release()


def audio_callback(input, frames, time, stat):
	q.put(input.ravel())


while True:
	data = ser.read_until("\r\n")
	try:
		fix = pynmea2.parse(data[7:])
		print(fix)
		if valid == True:
			break
		if fix.is_valid:
			print("Acquired GPS location\n")
			gps_coords = (fix.lat, fix.lon)
			valid = True
	except pynmea2.nmea.ParseError:
		pass

t1 = threading.Thread(target=gps_fix)

t1.start()

with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint("./Model-saved-2-1500-iter"))
	stream = sd.InputStream(blocksize=11025, dtype='int16', callback=audio_callback)
	stream.start()

	while True:
		if (time.time() - t_now1) >= 10.0:
			counter = 0
			t_now1 = time.time()
		if counter >= 3:
			wiringpi.digitalWrite(1, wiringpi.HIGH)
			text_mutex.acquire()
			try:
				ser.write("AT+GPSRD=0\r\n")
				time.sleep(0.1)
				ser.write("AT+CMGF=1\r\n")
				time.sleep(0.1)
				ser.write("AT+CMGS=+962798334636\r\n")
				time.sleep(0.1)
				ser.write("The sender is possibly being attacked by a dog\r\n")
				time.sleep(0.1)
				ser.write("The sender's last recorded location is " + gps_coords[0] + " " + gps_coords[1] + "\r\n")
				time.sleep(0.1)
				ser.write(chr(26) + "\r\n")
				time.sleep(0.1)
				ser.write("AT+GPSRD=1\r\n")
			finally:
				text_mutex.release()
				time.sleep(10)
				wiringpi.digitalWrite(1, wiringpi.LOW)
				counter = 0
		if not q.empty():
			spec = util.getSpect(q.get(), 44100)
			eval_res = myModel.y_conv.eval(session=sess, feed_dict={myModel.x : spec, myModel.keep_prob : 1.0})
                        print(eval_res)
			if eval_res.item(0) >= 0.5:
				counter += 1
