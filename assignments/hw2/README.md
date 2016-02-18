Make this simulator run in parallel on multicore using OpenMP, and GPU using CUDA, using the following sequence:

*   First comment out the image drawing and closest_approach calculation, and measure the raw asteroid-moving power.  You may need to add an asteroid output array to keep the compiler honest.

*   Next add the image drawing.  (On CUDA, I actually cleared the image using a single-threaded kernel<<<1,1>>>.)  Do you seem to need atomic operations or locks to make this produce a reasonable image?

*   Finally add the closest approach computation.  Because this number is important, your code should be guaranteed to always give the same result for this as the original sequential code.  (Note CUDA is missing an atomicMin for floats; I wrote the per-asteroid closest values to an array, and copied the whole array back to the CPU to take the global minimum.  The result is still off by one in the last digit due to different rounding.)

```cpp
#include <iostream>
#include <fstream>
#include "lib/inc.c" // netrun timing functions

// Make up a random 3D vector in this range.
//   NOT ACTUALLY RANDOM, just pseudorandom via linear congruence.
void randomize(int index,float range,float &x,float &y,float &z) {
    index=index^(index<<24); // fold index (improve whitening)
    x=(((index*1234567)%1039)/1000.0-0.5)*range;
    y=(((index*7654321)%1021)/1000.0-0.5)*range;
    z=(((index*1726354)%1027)/1000.0-0.5)*range;
}

class position {
public:
    float px,py,pz; // position's X, Y, Z components (meters)
    
    // Return distance to another position
    float distance(const position &p) const {
        float dx=p.px-px;
        float dy=p.py-py;
        float dz=p.pz-pz;
        return sqrt(dx*dx+dy*dy+dz*dz);
    }
};

class body : public position {
public:
    float m; // mass (Kg)
};

class asteroid : public body {
public:
    float vx,vy,vz; // velocity (m)
    float fx,fy,fz; // net force vector (N)
    
    void setup(void) {
        fx=fy=fz=0.0;
    }
    
    // Add the gravitational force on us due to this body
    void add_force(const body &b) {
        // Newton's law of gravitation:
        //   length of F = G m1 m2 / r^2
        //   direction of F = R/r
        float dx=b.px-px;
        float dy=b.py-py;
        float dz=b.pz-pz;
        float r=sqrt(dx*dx+dy*dy+dz*dz);
        
        float G=6.67408e-11; // gravitational constant
        float scale=G*b.m*m/(r*r*r);
        fx+=dx*scale;
        fy+=dy*scale;
        fz+=dz*scale;
    }
    
    // Use known net force values to advance by one timestep
    void step(float dt) {
        float ax=fx/m, ay=fy/m, az=fz/m;
        vx+=ax*dt; vy+=ay*dt; vz+=az*dt;
        px+=vx*dt; py+=vy*dt; pz+=vz*dt;
    }
};

// A simple fixed-size image
class image {
public:
    enum { pixels=500 };
    unsigned char pixel[pixels][pixels];
    void clear(void) {
        for (int y=0;y<pixels;y++)
            for (int x=0;x<pixels;x++) 
                pixel[y][x]=0;
    }
    
    void draw(float fx,float fy) {
        int y=(int)(fx*pixels);
        int x=(int)(fy*pixels);
        if (y>=0 && y<pixels && x>=0 && x<pixels)
            if (pixel[y][x]<200) pixel[y][x]+=10;
    }
    
    void write(const char *filename) {
        std::ofstream f("out.ppm",std::ios_base::binary);
        f<<"P5 "<<pixels<<" "<<pixels<<"\n";
        f<<"255\n";
        for (int y=0;y<pixels;y++)
            for (int x=0;x<pixels;x++) 
                f.write((char *)&pixel[y][x],1);
    }
};

int main(void) {
    image img;
    
    enum { n_asteroids=8192 };
    float range=500e6;
    float p2v=3.0e-6; // position (meters) to velocity (meters/sec)
    
    body terra; 
    terra.px=0.0; terra.py=0.0; terra.pz=0.0; 
    terra.m=5.972e24;
    
    body luna;
    luna.px=384.4e6; luna.py=0.0; luna.pz=0.0;
    luna.m=7.34767309e22;

    for (int test=0;test<5;test++) {
        float closest_approach=1.0e100;
        img.clear(); // black out the image
    
        double start=time_in_seconds();
/* performance critical part here */
        for (int ai=0;ai<n_asteroids;ai++)
        {
            asteroid a;
            int run=0;
            do {
                randomize(ai*100+run,range,a.px,a.py,a.pz);
                run++;
            } while (a.distance(terra)<10000e3);
            a.m=1.0;
            a.vx=-a.py*p2v; a.vy=a.px*p2v; a.vz=0.0;
            
            for (int i=0;i<1000;i++)
            {
                a.setup();
                a.add_force(terra);
                a.add_force(luna);
                a.step(1000.0);
                
                // Draw current location of asteroid
                img.draw(
                        a.px*(1.0/range)+0.5,
                        a.py*(1.0/range)+0.5);
                
                // Check distance
                float d=terra.distance(a);
                if (closest_approach>d) closest_approach=d;
            }
        }
        
        double elapsed=time_in_seconds()-start;
        std::cout<<"Took "<<elapsed<<" seconds, "<<elapsed*1.0e9/n_asteroids<<" ns/asteroid\n";
        std::cout<<"  closest approach: "<<closest_approach<<"\n";
    }

    img.write("out.ppm"); // netrun shows "out.ppm" by default
}
```