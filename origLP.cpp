/**
 * Local Plane with Savitzky Golay filter from https://www.frontiersin.org/articles/10.3389/fnins.2016.00176/full
 */

#include "origLP.hpp"

#include <stdlib.h>
#include <math.h>

origLP::origLP()
{
    total_evts = 0;
}

origLP::~origLP()
{
	delete[] on_event_cloud;
    delete[] off_event_cloud;
    // swfile.close();
}

void origLP::init()
{
	// swfile.open("sw_results.txt");

	// initialize local flow data arrays
    on_event_cloud = new int[frame_length];
    off_event_cloud = new int[frame_length];

    for (int i = 0; i < frame_length; ++i)
    {
        on_event_cloud[i] = 0;
        off_event_cloud[i] = 0;
    }
}

flow_event_t origLP::get_flow(event_t event)
{
	// add event to event cloud
    int last = 0;
    if (event.p == 1)
    {
        last = on_event_cloud[event.x * frame_width + event.y];
        on_event_cloud[event.x * frame_width + event.y] = event.t;
    }
    else
    {
        last = off_event_cloud[event.x * frame_width + event.y];
        off_event_cloud[event.x * frame_width + event.y] = event.t;
    }

    // check if pixel has been inactive for at least t_refract
    // if (last + refractTime <= event.t)
    // {
        // compute local, normal flow
        flow_event_t lf = _compute_local_plane(event);
        total_evts++;
        if (lf.mag != 0)
        {
            return lf;
        }
        else
            return flow_event_t(event, 0, 0, 0);
    // }
    // else
    //     return flow_event_t(event, 0, 0, 0);
}


flow_event_t origLP::_compute_local_plane(event_t event)
{
	int *event_cloud;
    if (event.p == 1)
        event_cloud = on_event_cloud;
    else
        event_cloud = off_event_cloud;

    float a10 = 0;
    float a01 = 0;
    int ii = 0;
    int jj = 0;

    for (int i = -searchDistance; i <= searchDistance; ++i)
    {
        for (int j = -searchDistance; j <= searchDistance; ++j)
        {
            if (event.x + i >= 0 && event.x + i < frame_height && event.y + j >= 0 && event.y + j < frame_width) // within bounds
            {
                int t1 = event_cloud[(event.x + i) * frame_width + (event.y + j)]; // pixel in frame
                if (t1 != 0 && event.t - t1 < maxDtThreshold)
                {
                    for (int xx = i + 1; xx <= searchDistance; ++xx)
                    {
                        if (event.x + xx >= 0 && event.x + xx < frame_height)
                        {
                            int t2 = event_cloud[(event.x + xx) * frame_width + (event.y + j)];
                            if (t2 != 0 && event.t - t2 < maxDtThreshold)
                            {
                                a10 += (float)(t2 - t1) / (xx - i);
                                ii++;
                            }
                        }
                    }
                    for (int yy = j + 1; yy <= searchDistance; ++yy)
                    {
                        if (event.y + yy >= 0 && event.y + yy < frame_width)
                        {
                            int t2 = event_cloud[(event.x + i) * frame_width + (event.y + yy)];
                            if (t2 != 0 && event.t - t2 < maxDtThreshold)
                            {
                                a01 += (float)(t2 - t1) / (yy - j);
                                jj++;
                            }
                        }
                    }
                }
            }
        }
    }

    swfile << a10 << " " << a01 << std::endl;

    if (ii < cut || jj < cut)
        return flow_event_t(event, 0, 0, 0);
    else
    {
        a10 /= ii;
        a01 /= jj;
    }

    a10 *= 1e-6;
    a01 *= 1e-6;

    if (std::abs(a10) < thr && std::abs(a01) < thr)
        return flow_event_t(event, 0, 0, 0);
    else
    {
        float temp = 1.0 / (a10 * a10 + a01 * a01);
        float vx = a10 * temp;
        float vy = a01 * temp;
        float mag = std::round(std::sqrt(vx * vx + vy * vy));
        return flow_event_t(event, vx, vy, mag);
    }
}