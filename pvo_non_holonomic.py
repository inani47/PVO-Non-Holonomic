import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math
from IPython import display


r_rad = 0.5
obs_rad = 0.5
r_pos = np.array([1.0,1.0])
r_vel = np.array([0.0,0.0]) #v,omega
r_heading = (math.pi)/4
linear_vel = np.array([r_vel[0]*math.cos(r_heading), r_vel[0]*math.sin(r_heading)])
goal = np.array([10.0,10.0])
obst_pos = np.array([7.5,7.5])
obst_pos1 = np.array([7.5,16])

obst_vel = np.array([-1.0, -1.0])
obst_vel1 = np.array([0.0, -0.75])
dt = 0.1
v_max=  1.0
v_min = .2
w_cap = 0.5

goal_heading = math.atan2(goal[1]-r_pos[1],goal[0]-r_pos[0])

print(goal_heading)
rel_pos = obst_pos - r_pos
rel_vel = obst_vel - linear_vel

rel_pos1 = obst_pos1 - r_pos
rel_vel1 = obst_vel1 - linear_vel
print(type(rel_vel))

mu_x = 0
std_x = 0.2

mu_y = 0
std_y = 0.2

n_samples = 100

def taylor_approx(v,w,dt,a,r1,r2,r,R,head):
    t0=(v**2)*(r1*math.cos(head + a*dt) +r2*math.sin(head + a*dt))**2 + (v**2)*((r+R)**2 - (r1**2 + r2**2) )
    d = 2*(v**2)*(dt)*(r1*math.cos((head + a*dt)) + r2*math.sin(head + a*dt))*(-r1*math.sin(head + a*dt) + r2*math.cos(head + a*dt))*(w-a)
    return (t0 + d)


def cost_fn(u, v_des, heading_des, v_cur, heading_cur, current_rel_p,current_rel_v,agent_r, obs_r,agent_v,dt, noise ):
    # Store the list of function calls
    return (v_des - (v_cur[0] + u[0]))**2 + (heading_des - (heading_cur + (v_cur[1] + u[1])*dt)) **2

def vel_cons(u, v_des, heading_des, v_cur, heading_cur, current_rel_p,current_rel_v,agent_r, obs_r,agent_v,dt, noise ):
    # Store the list of function calls
    return u[0]
def omega_cons(u, v_des, heading_des, v_cur, heading_cur, current_rel_p,current_rel_v,agent_r, obs_r,agent_v,dt, noise ):
    # Store the list of function calls
    return abs(u[1]) + 0.5

def ctrl_cons1(u, v_des, heading_des, v_cur, heading_cur, current_rel_p,current_rel_v,agent_r, obs_r,agent_v,dt, noise ):
    # Store the list of function calls
    return -(1 + agent_v + u[0])


def obt_avoidance_constraint(u, v_des, heading_des, v_cur, heading_cur, current_rel_p,current_rel_v,agent_r, obs_r,agent_v,dt, noise):

    if (np.linalg.norm(current_rel_p)>6):
        return 0

    noisy_rel_p = current_rel_p.reshape(-1,1) + noise
    
    v0=(agent_v[0])*np.array([math.cos(0), math.sin(0)])
    v1=(agent_v[0] + u[0]) *np.array([math.cos((agent_v[1] + u[1])*dt), math.sin((agent_v[1] + u[1])*dt)])
    fut_rel_v = (current_rel_v + v0 -v1)
    linear_vel = np.linalg.norm(fut_rel_v) #math.sqrt(noisy_rel_p[:][0]**2 + noisy_rel_p[:][1]**2) 
    new_head = math.atan2(fut_rel_v[1],fut_rel_v[0])
    a = math.atan2(current_rel_v[1],current_rel_v[0])

    if (a>3 and new_head<0):
        new_head =new_head +2*math.pi 
    elif(a<(-3) and new_head>0 ) :
        new_head = new_head-2*math.pi
    ang_vel = (new_head-a)/dt
    r = noisy_rel_p
    t = taylor_approx(linear_vel,ang_vel,dt,0,r[0][:],r[1][:],obs_r,agent_r,a)
    s= np.std(t)

    m = np.mean(t)
    return -(m + 1*s)
    
u_guess = np.array([0, 0])
n = 1 
while np.linalg.norm(r_pos - goal) > 1.0:
    v_desired = v_max
    w_desired = goal_heading
    v_current = r_vel[0]
    w_current = r_vel[1]

    transform = np.array([[math.cos(r_heading), math.sin(r_heading)], 
                         [- math.sin(r_heading), math.cos(r_heading)]])

    rel_pos = np.transpose(np.matmul(transform,np.transpose(rel_pos)))
    rel_vel = np.transpose(np.matmul(transform,np.transpose(rel_vel)))

    rel_pos1 = np.transpose(np.matmul(transform,np.transpose(rel_pos1)))
    rel_vel1 = np.transpose(np.matmul(transform,np.transpose(rel_vel1)))

    noise = (np.array([np.random.normal(mu_x, std_x, n_samples), np.random.normal(mu_y, std_y,n_samples) ]))
    
    


    arguments = (
    v_desired,
    w_desired,
    r_vel,
    r_heading,
    rel_pos,
    rel_vel,
    r_rad,
    obs_rad,
    r_vel,
    dt,
    noise
)

    arguments1 = (
        v_desired,
        w_desired,
        r_vel,
        r_heading,
        rel_pos1,
        rel_vel1,
        r_rad,
        obs_rad,
        r_vel,
        dt,
        noise
    )
    cons = ({'type': 'ineq', 'fun': obt_avoidance_constraint , 'args': arguments},
            {'type': 'ineq', 'fun': vel_cons , 'args': arguments},
            {'type': 'ineq', 'fun': omega_cons , 'args': arguments},
             {'type': 'ineq', 'fun': obt_avoidance_constraint , 'args': arguments1})

    #if (np.linalg.norm(rel_pos)<10):
    sol = optimize.minimize(cost_fn, u_guess, constraints = cons, args=arguments)
    #else:
    #    sol = optimize.minimize(cost_fn, np.array([0, 0]), method="SLSQP", args=arguments
     #               )
    u_sol = sol['x']
    u_guess = u_sol
    # print(sol)
    transform_back = np.array([[math.cos(r_heading), -math.sin(r_heading)], 
                         [math.sin(r_heading), math.cos(r_heading)]])

    rel_pos = np.transpose(np.matmul(transform_back,np.transpose(rel_pos)))
    rel_vel = np.transpose(np.matmul(transform_back,np.transpose(rel_vel)))

    rel_pos1 = np.transpose(np.matmul(transform_back,np.transpose(rel_pos1)))
    rel_vel1 = np.transpose(np.matmul(transform_back,np.transpose(rel_vel1)))


    r_vel =r_vel + np.array([u_sol[0], u_sol[1]])

    if r_vel[0]>v_max:
        r_vel[0] = v_max
    
    if r_vel[1]>w_cap:
        r_vel[1] = w_cap
    
    if r_vel[0]< v_min:
        r_vel[0] = v_min
    
    if r_vel[1] < -w_cap:
        r_vel[1] = -w_cap
    



    r_heading = r_heading + r_vel[1]*dt
    r_vel_2d = np.array([r_vel[0]*math.cos(r_heading), r_vel[0]*math.sin(r_heading)])
    
    r_pos = r_pos + r_vel_2d*dt
    obst_pos = obst_pos + obst_vel*dt
    obst_pos1 = obst_pos1 + obst_vel1*dt
    rel_pos = obst_pos - r_pos
    rel_pos1 = obst_pos1 - r_pos
    goal_heading = math.atan2(goal[1]-r_pos[1],goal[0]-r_pos[0])

    rel_vel = obst_vel - r_vel_2d
    rel_vel1 = obst_vel1 - r_vel_2d
    

    circle1 = plt.Circle((r_pos[0], r_pos[1]), 0.5, color='b', zorder=1)
    circle2 = plt.Circle((obst_pos[0], obst_pos[1]), 0.5, color='r',zorder=1)
    circle3 = plt.Circle((obst_pos1[0], obst_pos1[1]), 0.5, color='r',zorder=1)
    

    

    goal_plt = plt.Circle((goal[0], goal[0]), 0.1, color='g')
    

    
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    for i in range(25):
        ax.add_patch(plt.Circle((obst_pos[0]+ noise[0][i], obst_pos[1]+noise[1][i]), 0.5, color='r', alpha=0.1, zorder=2))
    
    for i in range(25):
        ax.add_patch(plt.Circle((obst_pos1[0]+ noise[0][i], obst_pos1[1]+noise[1][i]), 0.5, color='r', alpha=0.1, zorder=2))

    ax.add_patch(goal_plt)
    ax.arrow(r_pos[0],r_pos[1], math.cos(r_heading),  math.sin(r_heading), head_width = 0.2, width = 0.05)
    
    
    plt.pause(0.05)
    display.clear_output(wait=True)
    display.display(plt.gcf())