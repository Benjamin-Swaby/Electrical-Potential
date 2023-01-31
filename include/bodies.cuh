#pragma once


namespace bodies {

    struct pVec {
        double x;
        double y;
        double z;
    };

    typedef struct pVec pVec;

    struct body {
        pVec position;
        double charge;
        
        body(){}
        ~body(){}

        inline body(double c, double x, double y, double z) {
            this->charge = c;
            this->position.x = x;
            this->position.y = y;
            this->position.z = z;
        }

    };

    typedef struct body body;
}