import numpy as np
import gymnasium as gym

class mmWaveEnv(gym.Env):
    def __init__(self):
        self.Horizon = int(3600/5)                     # 1 hour divided into 5 seconds time slot.
        
        self.road_width = 7                            # in meters
        self.road_length = 100                         # in meters
        self.bs_location = np.array([[50,7],[20,0]])   # in meters
        
        self.Nbeams = 10                               # Number of mmWave beams
        self.beam_angle = int(180/self.Nbeams)         # in degrees
        
        # Computing the beam directions
        self.beam_directions = self.compute_beam_directions(self.beam_angle)
        
        self.bs_response_time = 0.5    # in seconds.
        
        self.Ncars_max = 4
        self.num_car_arr = np.arange(1, self.Ncars_max+1, dtype=int)
        self.num_car_prob = np.array([0.2, 0.4, 0.3, 0.1])
        
        self.car_direction_prob = np.array([[0.5, 0.5],[0.2,0.8],[0.8,0.2]])
        self.max_speed = 50  # in kmph
        self.velocity_bins = np.array([[0,20],[20,40],[40,self.max_speed]]) # in kmph
        self.Nv_bins = len(self.velocity_bins)
        self.velocity_bins_ix = np.arange(self.Nv_bins)
        self.velocity_prob = np.array([0.25, 0.5, 0.25])
        
        r_max1 = np.sqrt((self.bs_location[0,0]+30)**2+self.road_width**2)
        r_max2 = np.sqrt((self.road_length+30-self.bs_location[1,0])**2+self.road_width**2)
        r_max = max(r_max1, r_max2)
        self.power_bias = 2*np.log(r_max)
        
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(self.Nbeams)])  # Action space
        
        self.delta = self.max_speed*(1000/3600)*self.bs_response_time
        obsv_space = gym.spaces.Box(np.array([-self.delta, 0, -self.max_speed]), np.array([self.delta+self.road_length, self.road_width, self.max_speed]))
        self.observation_space = gym.spaces.Sequence(obsv_space)
        
        self.xpos_bins = np.array([[-self.delta,30],[30,70],[70,self.road_width+self.delta]])
        self.Nx_bins = len(self.xpos_bins)
        self.xpos_bins_ix = np.arange(self.Nx_bins)
        self.xpos_prob = np.array([0.25, 0.6, 0.15])
                
        self.mobile_blockage_prob = 0.15
        self.fixed_blockage = [[[65, 75], [10, 15]],[[50 ,55], [80, 90]]]
        
        self.Ncars = None              # Number of cars who wants to communicate in current time slot.
        self.observation = None        # Used to store the current observation.
        self.car_position = None       # Used to store the current position of the cars.
        self.t = None                  # Current time slot.
        
        
    def step(self, action):
        assert self.observation is not None, "Call reset before using step method!"
        assert self.t<self.Horizon, "The number of time slots exceeds the time horizon!"

        reward = self.reward_func(action)
        
        info = {}
        terminated = False
        self.t+=1
        if self.t>=self.Horizon:
            truncated = True
            return tuple(), reward, terminated, truncated, {}
        else:
            truncated = False
            self.generate_observation()
            return self.observation, reward, terminated, truncated, info
    
    
    def reset(self):
        self.t = 0
        self.generate_observation()
        info = {}
        return self.observation, info
    
    
    def render(self):
        pass
        
    
    def compute_beam_directions(self, beam_angle):
        return np.array([(180-beam_angle)*n/(self.Nbeams-1)+(beam_angle/2) for n in range(self.Nbeams)])
    
    
    def generate_observation(self):        
        self.Ncars = np.random.choice(self.num_car_arr, p=self.num_car_prob)
        self.observation = np.zeros((self.Ncars, 3))
        self.car_position = np.zeros((self.Ncars, 2))       
        
        for ix_car in range(self.Ncars):
            # x-coordinates of the car
            x_ix = np.random.choice(self.xpos_bins_ix, p=self.xpos_prob)
            x_position = (self.xpos_bins[x_ix][1]-self.xpos_bins[x_ix][0])*np.random.uniform()+self.xpos_bins[x_ix][0]
            
            # direction of the car
            if x_position>=self.xpos_bins[1,0] and x_position<=self.xpos_bins[1,1]:
                if x_position<=0.5*(self.xpos_bins[1,0]+self.xpos_bins[1,1]):
                    dir_cur = np.random.choice([-1,1], p=self.car_direction_prob[1])
                else:
                    dir_cur = np.random.choice([-1,1], p=self.car_direction_prob[2])
            else:
                dir_cur = np.random.choice([-1,1], p=self.car_direction_prob[0])
        
            # y-coordinates of the car
            if dir_cur==-1:
                mid = (self.road_width/2)/2        
            else:
                mid = (self.road_width/2)/2 + (self.road_width/2)
            
            delta = (self.road_width/2)/4
            y_position = np.random.uniform(low=mid-delta, high=mid+delta)
            
            delta_r = np.random.uniform(0, 2)
            delta_theta = np.random.uniform(0, 2*np.pi)
            x_meas = x_position + delta_r*np.cos(delta_theta)
            y_meas = y_position + delta_r*np.sin(delta_theta)
            x_meas = np.clip(x_meas, -self.delta, self.road_length+self.delta)
            if dir_cur==-1:
                y_meas = np.clip(y_meas, 0, self.road_width/2)
            else:
                y_meas = np.clip(y_meas, self.road_width/2, self.road_width)
            
            self.observation[ix_car, 1] = y_meas
            self.observation[ix_car, 0] = x_meas
            
            # Velocity of the car
            v_ix = np.random.choice(self.velocity_bins_ix, p=self.velocity_prob)
            velocity = (self.velocity_bins[v_ix][1]-self.velocity_bins[v_ix][0])*np.random.uniform()+self.velocity_bins[v_ix][0]
            delta_v = np.random.uniform(0, 5)
            velocity_meas = np.clip(velocity + delta_v, 0, self.max_speed)
            self.observation[ix_car, 2] = velocity_meas*dir_cur
            
            # Actual position of car during transmission
            angle = np.random.uniform(-15*np.pi/180.0, 15*np.pi/180.0)
            x_pos_cur = x_position+velocity*np.cos(angle)*dir_cur*(1000/3600)*self.bs_response_time
            y_pos_cur = y_position+velocity*np.sin(angle)*dir_cur*(1000/3600)*self.bs_response_time
            self.car_position[ix_car, 0] = np.clip(x_pos_cur, -self.delta, self.road_length+self.delta)
            if dir_cur==-1:
                self.car_position[ix_car, 1] = np.clip(y_pos_cur, 0.02*self.road_width/2, 0.98*self.road_width/2)
            else:
                self.car_position[ix_car, 1] = np.clip(y_pos_cur, 1.02*self.road_width/2, self.road_width/2+0.98*self.road_width/2)
            
        self.observation = tuple(self.observation)
    
    
    def reward_func(self, action):
        bs_ix = action[0]
        beam_number = action[1]
        
        angle_bounds = np.zeros(2)
        angle_bounds[0] = self.beam_directions[beam_number] - self.beam_angle/2
        angle_bounds[1] = self.beam_directions[beam_number] + self.beam_angle/2
        
        reward = 0
        
        for ix_car in range(self.Ncars):
            if bs_ix==0:
                angle = (180/np.pi)*np.arctan2(self.bs_location[bs_ix, 1]-self.car_position[ix_car, 1], self.bs_location[bs_ix, 0]-self.car_position[ix_car, 0])
            else:
                angle = (180/np.pi)*np.arctan2(self.car_position[ix_car, 1]-self.bs_location[bs_ix, 1], self.car_position[ix_car, 0]-self.bs_location[bs_ix, 0])
        
            if angle>=angle_bounds[0] and angle<=angle_bounds[1]:
                r = np.sqrt((self.bs_location[bs_ix, 0]-self.car_position[ix_car, 0])**2+(self.bs_location[bs_ix, 1]-self.car_position[ix_car, 1])**2)
                power = -2*np.log(r)+self.power_bias
            
                if np.random.uniform()>=self.mobile_blockage_prob:
                    not_blocked = True
                    for bound in self.fixed_blockage[bs_ix]:
                        if self.car_position[ix_car, 0]>=bound[0] and self.car_position[ix_car, 0]<=bound[1]:
                            not_blocked = False
                            break
                    
                    if not_blocked:
                        reward+=power
        
        return reward