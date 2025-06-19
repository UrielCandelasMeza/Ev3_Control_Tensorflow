from ev3dev2.motor import MoveSteering, MoveDifferential, SpeedPercent, OUTPUT_A, OUTPUT_D
from ev3dev2.wheel import EV3Tire
from time import sleep

class Movement:
    def __init__(self):
        self.steering = MoveSteering(OUTPUT_A, OUTPUT_D)
        self.mdiff = MoveDifferential(OUTPUT_A, OUTPUT_D, EV3Tire, 170)
        #self.leftMotor = MediumMotor(OUTPUT_D)
        #self.rightMotor = MediumMotor(OUTPUT_A)

    def forward(self, speed=50):
        self.steering.on_for_seconds(steering=0, speed=-speed, seconds=2)
    
    def back(self, speed=50):
        self.steering.on_for_seconds(steering=0, speed=speed, seconds=2)

    def turn_left(self, speed=30):
        self.mdiff.turn_left(SpeedPercent(30), 45)        
    def turn_right(self, speed=30):
        self.mdiff.turn_right(SpeedPercent(30), 45)        

    def stop(self):
        self.steering.off()