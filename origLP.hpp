/*
* Local Plane with Savitzky Golay Implementation from https://www.frontiersin.org/articles/10.3389/fnins.2016.00176/full
*/

#ifndef ORIG_LOCAL_PLANE
#define ORIG_LOCAL_PLANE

#include <inttypes.h>
#include <iostream>
// #include "lp.hpp"

// Event structure for individual events
typedef struct event_t
{
    uint16_t x; // x position of event
    uint16_t y; // y position of event
    int t;      // event time in us
    int p;      // polarity of event (1 == on, 0 ==off)

    // event_t constructor
    event_t(){};
    event_t(uint16_t xset, uint16_t yset, uint32_t tset, bool pset)
        : x(xset), y(yset), t(tset), p(pset) {}

} event_t;

// structure for flow event
typedef struct flow_event_t
{
    event_t tc;
    float vx;
    float vy;
    float mag;

    // flow_t constructor
    flow_event_t(){};
    flow_event_t(event_t evt, float vxset, float vyset, float magset){
        tc = evt;
    	vx = vxset;
    	vy = vyset;
        mag = magset;
    };
} flow_event_t;

class origLP
{
    private:
        // local plane data
        int *on_event_cloud;
        int *off_event_cloud;

        int total_evts;

        std::ofstream swfile;

        // private functions
        flow_event_t _compute_local_plane(event_t evt);

    public:
        // constructor
        origLP(void);

        // destructor
        ~origLP();

        // initialize original local plane version
        void init(void);

        // calculate local flow
        flow_event_t get_flow(event_t event);
};

#endif