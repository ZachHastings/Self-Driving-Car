import RPi.GPIO as GPIO
import time

trig = 10
echo = 12

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)

def run():
    while True:
        GPIO.output(trig, False)
        time.sleep(1)

        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        while GPIO.input(echo)==0:
            pulseStart = time.time()

        while GPIO.input(echo)==1:
            pulseEnd = time.time()

        pulseDuration = pulseEnd - pulseStart

        distance = pulseDuration * 17150
        distance = round(distance, 2)

        if distance > 2 and distance < 400:
            print(distance)
        else:
            print("Out of range")
        
