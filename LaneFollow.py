from PID import PID

class LaneFollow():

    def __init__(self):
        #self.setpoint = 0
        self.pid = PID(10,1,1)
        self.upperLimit = 180
        self.lowerLimit = 1

    def calcAngle(self, processVariable):
        #print(processVariable)
        self.pid.update(processVariable)
        self.controlVariable = round(90.0+self.pid.output,2)
        if self.controlVariable > self.upperLimit:
            self.controlVariable = self.upperLimit
        elif self.controlVariable < self.lowerLimit:
            self.controlVariable = self.lowerLimit
        return self.controlVariable

        
