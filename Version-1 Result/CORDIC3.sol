// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract CORDIC
{
    int var1;

        function set(int x) public
    {
        var1 = x;
    }

    int64[20] public a = [int64(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];

    int64[20] public sin = [int64(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        
    int64[20] public cos = [int64(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];


    function arctan(int64 input) public view returns(int64)
    {   
        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = input;
        int64 ynew = input;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (y > 0)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            else if (y < 0)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            x = xnew;
            y = ynew;
        }
        return(ang);
    }

    function arcsin(int64 input) public view returns(int64)
    {
        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = 0;
        int64 ynew = 0;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (y < input)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            else if (y > input)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            x = xnew;
            y = ynew;
        }
        return(ang);
    }

    function arccos(int64 input) public view returns(int64)
    {
        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = 0;
        int64 ynew = 0;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (x < input)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            else if (x > input)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            x = xnew;
            y = ynew;
        }
        return(ang);
    }

    function fsin(int64 input) public view returns(int64)
    {
        if (input > 157079632)
        {
            input = 314159265 - input;
        }

        else if (input < -157079632)
        {
            input = -314159265 - input;
        }
        
        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = 0;
        int64 ynew = 0;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < input)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            else if (ang > input)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            x = xnew;
            y = ynew;
        }
        return(y);
    }

    function fcos(int64 input) public view returns(int64)
    {
        bool neg = false;
        
        if (input > 157079632)
        {
            input = 314159265 - input;
            neg = true;
        }

        else if (input < -157079632)
        {
            input = -314159265 - input;
            neg = true;
        }
        
        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = 0;
        int64 ynew = 0;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < input)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            else if (ang > input)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            x = xnew;
            y = ynew;
        }
        
        if(neg)
        {
            return(-x);
        }
        else
        {
            return(x);
        }
    }

    function ftan(int64 input) public view returns(int64)
    {
        if (input > 157079632)
        {
            input = input - 314159265;
        }

        else if (input < -157079632)
        {
            input = input + 314159265;
        }

        int64 x = 100000000;
        int64 xnew = 100000000;
        int64 y = 0;
        int64 ynew = 0;
        int64 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < input)
            {
                xnew = (x*cos[i] - y*sin[i]) / 100000000;
                ynew = (y*cos[i] + x*sin[i]) / 100000000;
                ang = ang + a[i];
            }

            else if (ang > input)
            {
                xnew = (x*cos[i] + y*sin[i]) / 100000000;
                ynew = (y*cos[i] - x*sin[i]) / 100000000;
                ang = ang - a[i];
            }

            x = xnew;
            y = ynew;
        }
        //now divide y and x
        int64 res; res = 0;
        int64 digit;
        int8 j;

        j = 1;
        while(j <= 8)
        {
            digit = y / x;
            res = res + digit;
            res = res * 10;
            y = (y % x)*10;
            j = j + 1;
        }

        return(res);
    }
}