import time

class PIDController:
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.Kp = 10000.0
        self.Ki = 750.0
        self.Kd = 8000.0
        self.bias = 0.0
        self.cum_e = 0.0
        self.prev_e = 0.0
        self.prev_time = time.time()
        return

    def reset(self):
        return

#TODO: Complete your PID control within this function. At the moment, it holds
#      only the bias. Your final solution must use the error between the
#      target_pos and the ball position, plus the PID gains. You cannot
#      use the bias in your final answer.
    def get_fan_rpm(self, vertical_ball_position):
        e = self.target_pos - vertical_ball_position
        now = time.time()
        p = self.Kp * e
        d = self.Kd * ((e - self.prev_e)/(now - self.prev_time))
        self.cum_e += e
        i = self.Ki * self.cum_e
        self.prev_e = e
        self.prev_time = now
        #threshold
        if d < -3000:
            d = -3000
        if d > 3000:
            d = 3000
        if i < -3000:
            i = -3000
        if i > 3000:
            i = 3000
        output = d + i + p
        return output

