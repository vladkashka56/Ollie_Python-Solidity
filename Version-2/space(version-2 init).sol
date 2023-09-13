// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

library MathLib {

    uint8 constant Decimals = 8;
    int256 constant PI = 314159265;

    function sqrt(uint256 y) internal pure returns (uint256) {
      uint256 result;
      if (y > 3) {
          result = y;
          uint256 x = y / 2 + 1;
          while (x < result) {
              result = x;
              x = (y / x + x) / 2;
          }
      } else if (y != 0) {
          result = 1;
      }

      return result;
    }

    function abs(int256 x) internal pure returns (int256) {
        return x < 0 ? x*(-1) : x;
    }

    /**
     * Return the sine of an integer approximated angle as a signed 10^8
     * integer.
     *
     * @param input A 14-bit angle. This divides the circle into 628318530(2*PI)
     *               angle units, instead of the standard 360 degrees.
     * @return The sine result as a number in the range -10^8 to 10^8.
     */
    function sin(int256 input) internal pure  returns(int256)
    {
        int256[20] memory arctan_table = [int256(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory sin_table = [int256(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory cos_table = [int256(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];

        while(input < 0) {
            input = input + 2*PI;
        }
        int256 _angle = int(input % 628318530);

        if (_angle > PI/2 && _angle <= PI) {
            _angle = PI - _angle;
        } else if(_angle > PI && _angle < PI*3/2) {
            _angle = PI - _angle;
        } else if(_angle >= PI*3/2 && _angle < 2*PI) {
            _angle = _angle - 2*PI;
        }

        int256 x = 100000000;
        int256 xnew = 100000000;
        int256 y = 0;
        int256 ynew = 0;
        int256 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < _angle)
            {
                xnew = (x*cos_table[i] - y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] + x*sin_table[i]) / 100000000;
                ang = ang + arctan_table[i];
            }

            else if (ang > _angle)
            {
                xnew = (x*cos_table[i] + y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] - x*sin_table[i]) / 100000000;
                ang = ang - arctan_table[i];
            }

            x = xnew;
            y = ynew;
        }
        return(y);
    }

    /**
     * Return the cos of an integer approximated angle as a signed 10^8
     * integer.
     *
     * @param input A 10^8 radian angle. This divides the circle into 628318530(2*PI)
     *               angle units, instead of the standard 360 degrees.
     * @return The cos result as a number in the range -10^8 to 10^8.
     */
    function cos(int256 input) internal pure returns(int256)
    {

        bool neg = false;
        int256[20] memory arctan_table = [int256(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory sin_table = [int256(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory cos_table = [int256(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];

        while(input < 0) {
            input = input + 2*PI;
        }
        int256 _angle = int(input % 628318530);

        if (_angle > PI/2 && _angle <= PI) {
            _angle = PI - _angle;
            neg = true;
        } else if(_angle > PI && _angle < PI*3/2) {
            _angle = _angle - PI;
            neg = true;
        } else if(_angle >= PI*3/2 && _angle < 2*PI) {
            _angle = 2*PI - _angle;
        }

        int256 x = 100000000;
        int256 xnew = 100000000;
        int256 y = 0;
        int256 ynew = 0;
        int256 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < _angle)
            {
                xnew = (x*cos_table[i] - y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] + x*sin_table[i]) / 100000000;
                ang = ang + arctan_table[i];
            }

            else if (ang > _angle)
            {
                xnew = (x*cos_table[i] + y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] - x*sin_table[i]) / 100000000;
                ang = ang - arctan_table[i];
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

    /**
     * Return the tan of an integer approximated angle as a signed 10^8
     * integer.
     *
     * @param input A 10^8 radian angle. This divides the circle into 628318530(2*PI)
     *               angle units, instead of the standard 360 degrees.
     * @return The tan result as a number in the range -10^8 to 10^8.
     */
    function tan(int256 input) internal pure returns(int256)
    {
        // int256 PI = 314159265;
        int256[20] memory arctan_table = [int256(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory sin_table = [int256(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory cos_table = [int256(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];

        while(input < 0) {
            input = input + 2*PI;
        }
        int256 _angle = int(input % 628318530);

        if (_angle >= PI/2 && _angle <= PI*3/2) {
            _angle = _angle - PI;
        } else if (_angle > 3*PI/2) {
            _angle = _angle - 3*PI/2;
        }

        int256 x = 100000000;
        int256 xnew = 100000000;
        int256 y = 0;
        int256 ynew = 0;
        int256 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (ang < _angle)
            {
                xnew = (x*cos_table[i] - y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] + x*sin_table[i]) / 100000000;
                ang = ang + arctan_table[i];
            }

            else if (ang > _angle)
            {
                xnew = (x*cos_table[i] + y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] - x*sin_table[i]) / 100000000;
                ang = ang - arctan_table[i];
            }

            x = xnew;
            y = ynew;
        }
        //now divide y and x
        int256 res; res = 0;
        int256 digit;
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

    /**
     * Return the angle of an integer approximated ratio as a signed 256-bit
     * integer.
     *
     * @param input A 256-bit(unit 1) ratio as a number in the range 0 to infinite.
     * @return The arcsin result is A 256-bit angle. This divides the circle into 4 * 340282366920938463463374607431768211455 (2^256)
     *               angle units, instead of the standard 360 degrees.                785118600829010179644344535919513282844767027200
     */
    function arcsin(int256 input) internal pure returns(int256)
    {
        int256[20] memory arctan_table = [int256(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory sin_table = [int256(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory cos_table = [int256(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];

        int256 x = 100000000;
        int256 xnew = 100000000;
        int256 y = 0;
        int256 ynew = 0;
        int256 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (y < input)
            {
                xnew = (x*cos_table[i] - y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] + x*sin_table[i]) / 100000000;
                ang = ang + arctan_table[i];
            }

            else if (y > input)
            {
                xnew = (x*cos_table[i] + y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] - x*sin_table[i]) / 100000000;
                ang = ang - arctan_table[i];
            }

            x = xnew;
            y = ynew;
        }
        return(ang);
    }



    /**
     * Return the angle of an integer approximated ratio as a signed 256-bit
     * integer.
     *
     * @param input A 256-bit(unit 1) ratio as a number in the range 0 to infinite.
     * @return The arctan result is A 256-bit angle. This divides the circle into 4 * 340282366920938463463374607431768211455 (2^256)
     *               angle units, instead of the standard 360 degrees.                785118600829010179644344535919513282844767027200
     */
    function arctan(int256 input) internal pure returns(int256)
    {   
        int256[20] memory arctan_table = [int256(78539816), 39269908, 19634954, 9817477, 4908739, 2454369, 1227185, 613592, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory sin_table = [int256(70710678), 38268343, 19509032, 9801714, 4906767, 2454123, 1227154, 613588, 306796, 153398, 76699, 38350, 19175, 9587, 4794, 2397, 1198, 599, 300, 150];
        int256[20] memory cos_table = [int256(70710678), 92387953, 98078528, 99518473, 99879546, 99969882, 99992470, 99998118, 99999529, 99999882, 99999971, 99999993, 99999998, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000];

        int256 x = 100000000;
        int256 xnew = 100000000;
        int256 y = input;
        int256 ynew = input;
        int256 ang = 0;

        uint8 i = 0;
        for (i = 0; i <= 19; i = i + 1)
        {
            if (y > 0)
            {
                xnew = (x*cos_table[i] + y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] - x*sin_table[i]) / 100000000;
                ang = ang + arctan_table[i];
            }

            else if (y < 0)
            {
                xnew = (x*cos_table[i] - y*sin_table[i]) / 100000000;
                ynew = (y*cos_table[i] + x*sin_table[i]) / 100000000;
                ang = ang - arctan_table[i];
            }

            x = xnew;
            y = ynew;
        }
        return(ang);
    }


    function arctan2(int256 Y, int256 X) internal pure returns(int256) 
    {
        int256 alpha = arctan( abs(int256(10**Decimals) * Y / X));
        if( X < 0 && Y > 0) {
            alpha = alpha * int256(-1);
        } 
        if( X > 0 && Y < 0) {
            alpha = PI - alpha;
        } 
        if( X < 0 && Y < 0) {
            alpha = alpha - PI;
        } 

        return(alpha);

    } 


    /**
     * conversion from ecliptic cartesian coordinates to equatorial ones
     * integer.
     *
     */
    function cart_ecl2cart_eq(int256 cart_X, int256 cart_Y, int256 cart_Z, int256 epsilon) internal pure returns(int256, int256, int256)
    {
        int256 equat_X = cart_X;
        int256 equat_Y = ( cart_Y*cos(epsilon) - cart_Z*sin(epsilon) )/int256( 10**Decimals );
        int256 equat_Z = ( cart_Y*sin(epsilon) + cart_Z*cos(epsilon) )/int256( 10**Decimals );
        return(equat_X, equat_Y, equat_Z);
    }


    /**
     * Conversion from cartesian to spherical coordinates
     * integer.
     * x axis points towards the (alpha = 0, delta = 0) point and xy plain is the delta = 0 plane 
     *
     */
    function cart2sph(int256 cart_X, int256 cart_Y, int256 cart_Z) internal pure returns(int256, int256)
    {
        int256 norm = int256(sqrt(uint256(cart_X*cart_X + cart_Y*cart_Y + cart_Z*cart_Z)));
        int256 alpha = (arctan2(cart_Y, cart_X) + 2*PI) % (2*PI);
        int256 delta = arcsin(int256(10**Decimals) * cart_Z / norm);

        return(alpha, delta);
    }


    /**
     * Conversion from Equatorial to Horizontal coordinates (both spherical)
     * integer.
     * x axis points towards the (alpha = 0, delta = 0) point and xy plain is the delta = 0 plane 
     *
     */
    function sph_eq2sph_hor(int256 timeIndex, int256 RA, int256 DEC, int256 lat, int256 lon ) internal pure returns(int256, int256)
    {
        int256 temp;
        // calculating Greenwich Sideral Time
        int256 tu = timeIndex / ( 24 * 3600 );
        int256 GST = 2*PI * ((77905727 + 100273781 * tu) % int256( 10**Decimals )) / int256(10**Decimals);
        // calculating Local Sideral Time
        int256 LST = GST + lon;

        // calculating the hour angle
        int256 h = LST - RA;
        // calculating the altitudes (a) and the azimuths (A) using spherical triangles
        int256 delta = DEC;

        temp = sin(lat) * sin(delta)/(int256(10**Decimals)) + cos(lat) * cos(delta) * cos(h)/(int256(10**(Decimals*2)));
        int256 a = arcsin(temp);

        temp = int256(-1) * cos(delta) * cos(h) * sin(lat)/(int256(10**(Decimals*2))) + sin(delta) * cos(lat)/(int256(10**Decimals));
        int256 A = int256(-1) * arctan2(cos(delta)*sin(h)/(int256(10**Decimals)), temp);

        return(a, A);
    }

    function get_sky_positions(int256 pos_X, int256 pos_Y, int256 pos_Z, int256 obliquity, int256 obs_latitude_rad, int256 obs_longitude_rad ) internal pure returns(int256, int256)
    { 
        int256 RA;
        int256 DEC;
        int256 equat_X;
        int256 equat_Y;
        int256 equat_Z;
        int256 skyPos_a;
        int256 skyPos_A;


        (equat_X, equat_Y, equat_Z) = cart_ecl2cart_eq(pos_X, pos_Y, pos_Z, obliquity);
        (RA, DEC) = cart2sph(equat_X, equat_Y, equat_Z);

        (skyPos_a, skyPos_A) = sph_eq2sph_hor(0, RA, DEC, obs_latitude_rad, obs_longitude_rad);

        return(skyPos_a, skyPos_A);

    }
}

pragma solidity ^0.8.0;

/**
 * @dev String operations.
 */
library Strings {
    bytes16 private constant _HEX_SYMBOLS = "0123456789abcdef";

    /**
     * @dev Converts a `int256` to its ASCII `string` decimal representation.
     */
    function toString(int256 value) internal pure returns (string memory) {
        // Inspired by OraclizeAPI's implementation - MIT licence
        // https://github.com/oraclize/ethereum-api/blob/b42146b063c7d6ee1358846c198246239e9360e8/oraclizeAPI_0.4.25.sol
        bool neg = false;
        int256 temp;
        int256 abs_value;
        if (value == 0) {
            return "0";
        }
        if (value < 0) {
            neg = true;
            temp = int(0 - value);
        } else {
            temp = int(value);
        }
        abs_value = temp;

        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (abs_value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(abs_value % 10)));
            abs_value /= 10;
        }

        return neg? string(abi.encodePacked('-', string(buffer))) : string(buffer);
    }

}

struct Orbit2D {
    int256 a;
    int256 e;
    int256 M;
    int256 T;
    int256 n;
}

struct Orbit3D {
    int256 a;
    int256 e;
    int256 M;
    int256 T;
    int256 Omega;
    int256 omega;
    int256 I;
    int256[3][3] Euler_angle_transformation_matrix;
    Orbit2D orbit2d;
}

struct Orbit {
    int256 a0;       // a [AU] - semi-major axis (default 1)
    int256 e0;       // e [0-1] - eccentricity (default 0)
    int256 M0;       // M0 - mean anomaly at time zero (default 0; recommended to use J2000 epoch for Solar System)
    int256 T0;       // T [sec] - orbital period (default 1)
    int256 Omega0;   // Omega [rad] - longitude of the ascending node (default 0)
    int256 omega0;   // omega [rad] - argument of the pericenter (default 0)
    int256 I0;       // I [rad] - inclination (default 0)
    mapping (bytes => int256) roc_funcs;
    Orbit3D orbit3d;
}


pragma solidity ^0.8.0;

library Orbit2DFuns {
    uint8 constant Decimals = 8;
    int256 constant PI = 314159265;

    /**
     * Sets the mean angular velocity (n) for the class
     * Needs to be run every time the period (T) is set or updated
     */
    function _update_n( Orbit2D storage self ) internal {
        self.n = 100 * 2 * PI / self.T;     //100*self.n to correct closely
    }

    /**
     * To be used for changes in orbital parameters, e.g. perihelion shift
     */
    function update( Orbit2D storage self, string memory key, int256 value ) internal {
        if (keccak256(bytes(key)) == keccak256(bytes('a')))
            self.a = value;
        if (keccak256(bytes(key)) == keccak256(bytes('e')))
            self.e = value;
        if (keccak256(bytes(key)) == keccak256(bytes('M')))
            self.M = value;
        if (keccak256(bytes(key)) == keccak256(bytes('T'))) {
            self.T = value;
            self.n = 100 * 2 * PI / value;
        }
    }

    /**
     * Arguments: t [sec] - relative time from point zero (default J2000 epoch) 
     * Returns: radius [AU] and true anomaly [rad]
     */
    function get_rv( Orbit2D storage self, int256 t ) internal view returns (int256, int256) {
        int256 M = self.n * t/100  + self.M;  // actual mean anomaly
        int256 E;                         // actual eccentric anomaly
        int256 temp;                      // Temporary memory for deep computations

        // condition to check if fsolve is success.
        // fsolve_cond = (np.pi-2/E0) * (np.pi-2/E0) - 4*(np.pi*np.pi/4 - 2 - 2*M0/E0) : Python format
        int256 fsolve_cond = (PI- int(10**Decimals) * 2*int(10**Decimals)/self.e);
        fsolve_cond = fsolve_cond * fsolve_cond;
        temp = int(10**Decimals) * 2*int(10**Decimals)*M/self.e;
        temp = 4*(PI*PI/4 - int(10**Decimals) * 2*int(10**Decimals) - temp );
        fsolve_cond = fsolve_cond - temp;

        if(fsolve_cond > 0) {
            int256 E0 = ((PI- int(10**Decimals) * 2*int(10**Decimals)/self.e) + int(MathLib.sqrt(uint(fsolve_cond))))/2;
            int256 E1 = ((PI- int(10**Decimals) * 2*int(10**Decimals)/self.e) - int(MathLib.sqrt(uint(fsolve_cond))))/2;
            int256 E0_offset = E0-self.e*MathLib.sin(int(E0))/int(10**Decimals);
            int256 E1_offset = E1-self.e*MathLib.sin(int(E1))/int(10**Decimals);
            E = MathLib.abs(E0_offset) < MathLib.abs(E1_offset) ? E0:E1;

            for(uint i=0; i<5; i++) {
                temp = int(10**Decimals) * (E - self.e*MathLib.sin(int(E))/int(10**Decimals) - M);
                temp = E - temp/(int(10**Decimals) - MathLib.cos(int(E))/int(10**Decimals));
                E = temp;
            }

            // Radius (distance from baricenter) => r = self.a * ( 1.0 - self.e * np.cos( E ) ) : Python format
            int256 r = self.a * ( int(10**Decimals) - self.e * MathLib.cos(E)/int(10**Decimals) ) / (int(10**Decimals));   
            // True anomaly => v = 2.0 * np.arctan( np.sqrt( ( 1.0 + self.e ) / ( 1.0 - self.e ) ) * np.tan( E / 2.0 ) ) : Python format                            // radius (distance from baricenter)
            temp = int(MathLib.sqrt(uint( int(10**Decimals) * int(10**Decimals) * (int(10**Decimals) + self.e )/( int(10**Decimals) - self.e ))));
            int256 ratio = temp * MathLib.sin(int(E/2))/MathLib.cos(int(E/2));
            temp = int(MathLib.arctan(int(ratio)));            
            int256 v = 2 * temp  ;  // true anomaly
            return(r, v);
        } else {
          // fsolve is false.
          return(0, 0);
        }
    }


    /**
     * Same as get_rv() but returns the objects position in cartesian coordinates 
     * Returns: x, y [AU] - x-axis is in the direction of pericenter, y-axis is right-hand perpendicular
     */
    function get_xy( Orbit2D storage self, int256 t ) internal view returns (int256, int256) {
        int256 r; 
        int256 v;
        (r, v) = get_rv(self, t);

        int256 x = r * MathLib.cos(int(v))/int(10**Decimals);
        int256 y = r * MathLib.sin(int(v))/int(10**Decimals);

        return(x, y);

    }

}

pragma solidity ^0.8.0;

library Orbit3DFuns {

    uint8 constant Decimals = 8;
    int256 constant PI = 314159265;

    function _update_Euler_angle_transformation_matrix( Orbit3D storage self) internal {
        int256 cosO = MathLib.cos(int(self.Omega));
        int256 sinO = MathLib.sin(int(self.Omega));

        int256 cosI = MathLib.cos(int(self.I));
        int256 sinI = MathLib.sin(int(self.I));

        int256 coso = MathLib.cos(int(self.omega));
        int256 sino = MathLib.sin(int(self.omega));

        self.Euler_angle_transformation_matrix[0][0] = (int(10**Decimals)*cosO*coso- sinO*sino*cosI)/(int(10**Decimals)*int(10**Decimals));
        self.Euler_angle_transformation_matrix[0][1] = (-int(10**Decimals)*cosO*sino- sinO*coso*cosI)/(int(10**Decimals)*int(10**Decimals));
        self.Euler_angle_transformation_matrix[0][2] = sinO*sinI/int(10**Decimals);
        self.Euler_angle_transformation_matrix[1][0] = (int(10**Decimals)*sinO*coso + cosO*cosI*sino)/(int(10**Decimals)*int(10**Decimals));
        self.Euler_angle_transformation_matrix[1][1] = (cosO*coso*cosI -int(10**Decimals)*sinO*sino)/(int(10**Decimals)*int(10**Decimals));
        self.Euler_angle_transformation_matrix[1][2] = -cosO*sinI/int(10**Decimals);
        self.Euler_angle_transformation_matrix[2][0] = sinI*sino/int(10**Decimals);
        self.Euler_angle_transformation_matrix[2][1] = sinI*coso/int(10**Decimals);
        self.Euler_angle_transformation_matrix[2][2] = cosI;
    }

    /**
     * Updates class parameters
     * To be used for changes in orbital parameters, e.g. perihelion shift
     */
    function update( Orbit3D storage self, string memory key, int256 value ) internal {
        if (keccak256(bytes(key)) == keccak256(bytes('a')))
            self.a = value;
        if (keccak256(bytes(key)) == keccak256(bytes('e')))
            self.e = value;
        if (keccak256(bytes(key)) == keccak256(bytes('M')))
            self.M = value;
        if (keccak256(bytes(key)) == keccak256(bytes('T'))) 
            self.T = value;
        if (keccak256(bytes(key)) == keccak256(bytes('Omega'))) {
            self.Omega = value;
            _update_Euler_angle_transformation_matrix(self);
        }
        if (keccak256(bytes(key)) == keccak256(bytes('omega'))) {
            self.omega = value;
            _update_Euler_angle_transformation_matrix(self);
        }
        if (keccak256(bytes(key)) == keccak256(bytes('I'))) {
            self.I = value;
            _update_Euler_angle_transformation_matrix(self);
        }
        
        Orbit2DFuns.update(self.orbit2d, key, value);
    }

    /**
     * Argument: t [sec] - time since time zero [default to J2000 epoch]
     * Returns: objects possition x, y, z in [AU] - x is directed at vernal equinox; y,z are right-handed ortogonal
     */
    function get_xyz( Orbit3D storage self, int256 t) internal view returns (int256, int256, int256) {
        int256 p_2d_x;
        int256 p_2d_y;
        (p_2d_x, p_2d_y) = Orbit2DFuns.get_xy(self.orbit2d, t);

        int256 x = (self.Euler_angle_transformation_matrix[0][0] * p_2d_x + self.Euler_angle_transformation_matrix[0][1] * p_2d_y)/int(10**Decimals);
        int256 y = (self.Euler_angle_transformation_matrix[1][0] * p_2d_x + self.Euler_angle_transformation_matrix[1][1] * p_2d_y)/int(10**Decimals);
        int256 z = (self.Euler_angle_transformation_matrix[2][0] * p_2d_x + self.Euler_angle_transformation_matrix[2][1] * p_2d_y)/int(10**Decimals);

        return(x, y, z);

    }

    /**
     * Argument: t [sec] - time since time zero [default to J2000 epoch]
     * Returns: objects possition x, y in [AU] - x is directed at vernal equinox; y is right-handed ortogonal
     */
    function get_xy( Orbit3D storage self, int256 t) internal view returns (int256, int256) {
        int256 p_2d_x;
        int256 p_2d_y;
        (p_2d_x, p_2d_y) = Orbit2DFuns.get_xy(self.orbit2d, t);
        
        return(p_2d_x, p_2d_y);
    } 

}

pragma solidity ^0.8.0;

library OrbitFuns {

    uint8 constant Decimals = 8;
    int256 constant PI = 314159265;

    /**
     * Updates class parameters
     * To be used for changes in orbital parameters, e.g. perihelion shift
     */
    function update( Orbit storage self, string memory key, int256 value ) internal {
        if (keccak256(bytes(key)) == keccak256(bytes('a')))
            self.a0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('e')))
            self.e0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('M')))
            self.M0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('T'))) 
            self.T0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('Omega'))) 
            self.Omega0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('omega'))) 
            self.omega0 = value;
        if (keccak256(bytes(key)) == keccak256(bytes('I'))) 
            self.I0 = value;

        Orbit3DFuns.update(self.orbit3d, key, value);
    }


    /**
     * Argument: t [sec] - time since time zero [default to J2000 epoch]
     * Returns: objects possition x, y, z in [AU] - x is directed at vernal equinox; y,z are right-handed ortogonal
     */
    function get_xyz( Orbit storage self, int256 t) internal returns (int256, int256, int256) {
        string[7] memory params = ['a','e','M','T','Omega','omega','I'];
        for(uint i = 0; i < params.length; i++) {
            if( self.roc_funcs[bytes(params[i])] != 0) {
                int256 newVal;
                if (keccak256(bytes(params[i])) == keccak256(bytes('a')))
                    newVal = self.a0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('e')))
                    newVal = self.e0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('M')))
                    newVal = self.M0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('T'))) 
                    newVal = self.T0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('Omega')))
                    newVal = self.Omega0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('omega'))) 
                    newVal = self.omega0 + self.roc_funcs[bytes(params[i])] * t;
                if (keccak256(bytes(params[i])) == keccak256(bytes('I'))) 
                    newVal = self.I0 + self.roc_funcs[bytes(params[i])] * t;
                Orbit3DFuns.update(self.orbit3d, params[i], newVal);
            }
        }

        (int256 x, int256 y, int256 z) = Orbit3DFuns.get_xyz(self.orbit3d, t);

        return(x, y, z);

    }

}



pragma solidity ^0.8.0;
pragma abicoder v2;

contract Test {

    using Orbit2DFuns for Orbit2D;
    using Orbit3DFuns for Orbit3D;
    using OrbitFuns for Orbit;
    using Strings for int256;
    int256[3][3] public matrix;
    Orbit2D orbit2d;
    Orbit3D orbit3d;
    mapping (bytes => Orbit3D) private Solar_System_Keplerian_Elements;

    constructor() { 
        orbit2d =  Orbit2D({a:38709893, e:20563069, M: 305073760, T:760065407544000, n:8267});
        matrix = [[int256(21989544), int256(-97126040),int256(9110012)], [int256(97371598),int256(21284672),int256(-8107696)], [int256(5935648),int256(10653410),int256(99253579)]];
        orbit3d =  Orbit3D({a:38709893, e:20563069, M: 305073760, T:760065407544000, I:12225804, Omega:84354677, omega:50832330, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Mercury')] = orbit3d;
        
        orbit2d =  Orbit2D({a:72333199, e:677323, M: 88046188, T:1940826194496000, n:3237});
        matrix = [[int256(-66165487), int256(-74759120),int256(5762142)], [int256(74824365),int256(-66328376),int256(-1364163)], [int256(4841772),int256(3408881),int256(99824530)]];
        orbit3d =  Orbit3D({a:72333199, e:677323, M: 88046188, T:1940826194496000, I:12225804, Omega:133833051, omega:95735306, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Venus')] = orbit3d;
            
        orbit2d =  Orbit2D({a:100000011, e:1671022, M: -4333373, T:3155814950400000, n:1991});
        matrix = [[int256(-22405287), int256(-97457699),int256(0)], [int256( 97457699),int256(-22405287),int256(-87)], [int256(85),int256(-19),int256(100000000)]];
        orbit3d =  Orbit3D({a:100000011, e:1671022, M: -4333373, T:3155814950400000, I:12225804, Omega:0, omega:179676742, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Earth')] = orbit3d;

        orbit2d =  Orbit2D({a:152366231, e:9341233, M: 33881169, T:59360879217024000, n:106});
        matrix = [[int256(91345435), int256(40619790),int256(2458499)], [int256(-40576104),int256(91373931),int256(-2093935)], [int256(-3096979),int256(915151),int256(99947842)]];
        orbit3d =  Orbit3D({a:152366231, e:9341233, M: 33881169, T:59360879217024000, I:3229923, Omega:86530876, omega:499971031, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Mars')] = orbit3d;

        orbit2d =  Orbit2D({a:520336301, e:4839266, M: 34296644, T:37427965311744000, n:168});
        matrix = [[int256(96677441), int256(-25464826),int256(2239428)], [int256(25461952),int256(96703231),int256(417324)], [int256(-2271870),int256(166744),int256(99974051)]];
        orbit3d =  Orbit3D({a:520336301, e:4839266, M: 34296644, T:37427965311744000, I:2278178, Omega:175503590, omega:-149753264, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Jupiter')] = orbit3d;

        orbit2d =  Orbit2D({a:953707032, e:5415060, M: -74154886, T:92970308438784000, n:68});
        matrix = [[int256(-4274500), int256(-99829742),int256(3968795)], [int256(99896211),int256(-4208035),int256(1743422)], [int256(-1573445),int256(4039199),int256(99906002)]];
        orbit3d =  Orbit3D({a:953707032, e:5415060, M: -74154886, T:92970308438784000, I:4336200, Omega:198470185, omega:-37146017, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Saturn')] = orbit3d;

        orbit2d =  Orbit2D({a:1919126393, e:4716771, M: 248304397, T:265120013983104000, n:24});
        matrix = [[int256(-98750424), int256(-15706107),int256(1293045)], [int256(15702652),int256(-98758764),int256(-365167)], [int256(1334348),int256(-157561),int256(99990973)]];
        orbit3d =  Orbit3D({a:1919126393, e:4716771, M: 248304397, T:265120013983104000, I:1343659, Omega:129555580, omega:168833308, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Uranus')] = orbit3d;
        
        orbit2d =  Orbit2D({a:3006896348, e:858587, M: 453626222, T:520078303825920000, n:12});
        matrix = [[int256(70710505), int256(-70673294),int256(2304314)], [int256(70643638),int256(70747826),int256(2054634)], [int256(-3082329),int256(175009),int256(99952332)]];
        orbit3d =  Orbit3D({a:3006896348, e:858587, M: 453626222, T:520078303825920000, I:3087784, Omega:229897718, omega:-151407906, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Neptune')] = orbit3d;

        orbit2d =  Orbit2D({a:3948168677, e:24880766, M: 25939170, T:782957689194239900, n:8});
        matrix = [[int256(-68040055), int256(67870818),int256(27642411)], [int256(-68138896),int256(-72474109),int256(10227141)], [int256(26974835),int256(-11876681),int256(95557850)]];
        orbit3d =  Orbit3D({a:3948168677, e:24880766, M: 25939170, T:782957689194239900, I:29917997, Omega:192515872, omega:198554397, Euler_angle_transformation_matrix: matrix, orbit2d:orbit2d });
    //    orbit3d._update_Euler_angle_transformation_matrix();
        Solar_System_Keplerian_Elements[bytes('Pluto')] = orbit3d;
    }

    function get_relative_2D_pos(string memory object, string memory observer, int256 t ) public view returns (int256, int256) {
        int256 pos_x;
        int256 pos_y;
        int256 obs_x = 0;
        int256 obs_y = 0;
        // objective planet's absolute position.
        (pos_x, pos_y) = Solar_System_Keplerian_Elements[bytes(object)].get_xy(t);
        // observer planet's absolute position.
        (obs_x, obs_y) = Solar_System_Keplerian_Elements[bytes(observer)].get_xy(t);
        
        return (pos_x-obs_x, pos_y-obs_y);
    }

    function get_relative_all_2D_pos(string memory observer, int256 t ) public view returns (string[] memory) {
        int256 pos_x;
        int256 pos_y;
        int256 obs_x = 0;
        int256 obs_y = 0;
        string[9] memory planet_list = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'];
        string[] memory result_list = new string[](9);

        // observer planet's absolute position.
        (obs_x, obs_y) = Solar_System_Keplerian_Elements[bytes(observer)].get_xy(t);

        for (uint i=0; i< planet_list.length; i++) {
            // planet's absolute position.
            (pos_x, pos_y) = Solar_System_Keplerian_Elements[bytes(planet_list[i])].get_xy(t);
            result_list[i] = string(abi.encodePacked(planet_list[i], "_", (pos_x-obs_x).toString(), "_", (pos_y-obs_y).toString() ));
        }

        return result_list;
    }

    function get_absolutive_2D_pos(string memory object, int256 t ) public view returns (int256, int256) {
        int256 pos_x;
        int256 pos_y;

        // objective planet's absolute position.
        (pos_x, pos_y) = Solar_System_Keplerian_Elements[bytes(object)].get_xy(t);
        
        return (pos_x, pos_y);
    }

    function get_absolutive_all_2D_pos(int256 t ) public view returns (string[] memory) {
        int256 pos_x;
        int256 pos_y;
        string[9] memory planet_list = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'];
        string[] memory result_list = new string[](9);
        for (uint i=0; i< planet_list.length; i++) {
            // planet's absolute position.
            (pos_x, pos_y) = Solar_System_Keplerian_Elements[bytes(planet_list[i])].get_xy(t);
            result_list[i] = string(abi.encodePacked(planet_list[i], "_", pos_x.toString(), "_", pos_y.toString() ));
        }

        return result_list;
    }

    function get_relative_3D_pos(string memory object, string memory observer, int256 t ) public view returns (int256, int256, int256) {
        int256 pos_x;
        int256 pos_y;
        int256 pos_z;
        int256 obs_x = 0;
        int256 obs_y = 0;
        int256 obs_z = 0;
        
        // objective planet's absolute position.
        (pos_x, pos_y, pos_z) = Solar_System_Keplerian_Elements[bytes(object)].get_xyz(t);
        // observer planet's absolute position.
        (obs_x, obs_y, obs_z) = Solar_System_Keplerian_Elements[bytes(observer)].get_xyz(t);

        return (pos_x-obs_x, pos_y-obs_y, pos_z-obs_z);
    }

    function get_relative_all_3D_pos(string memory observer, int256 t ) public view returns (string[] memory) {
        int256 pos_x;
        int256 pos_y;
        int256 pos_z;
        int256 obs_x = 0;
        int256 obs_y = 0;
        int256 obs_z = 0;
        string[9] memory planet_list = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'];
        string[] memory result_list = new string[](9);

        // observer planet's absolute position.
        (obs_x, obs_y, obs_z) = Solar_System_Keplerian_Elements[bytes(observer)].get_xyz(t);

        for (uint i=0; i< planet_list.length; i++) {
            // planet's absolute position.
            (pos_x, pos_y, pos_z) = Solar_System_Keplerian_Elements[bytes(planet_list[i])].get_xyz(t);
            result_list[i] = string(abi.encodePacked(planet_list[i], "_", (pos_x-obs_x).toString(), "_", (pos_y-obs_y).toString(), "_", (pos_z-obs_z).toString() ));
        }

        return result_list;
    }


    function get_absolutive_3D_pos(string memory object, int256 t ) public view returns (int256, int256, int256) {
        int256 pos_x;
        int256 pos_y;
        int256 pos_z;

        // objective planet's absolute position.
        (pos_x, pos_y, pos_z) = Solar_System_Keplerian_Elements[bytes(object)].get_xyz(t);
        
        return (pos_x, pos_y, pos_z);
    }

    function get_absolutive_all_3D_pos(int256 t ) public view returns (string[] memory) {
        int256 pos_x;
        int256 pos_y;
        int256 pos_z;
        string[9] memory planet_list = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'];
        string[] memory result_list = new string[](9);

        for (uint i=0; i< planet_list.length; i++) {
            // planet's absolute position.
            (pos_x, pos_y, pos_z) = Solar_System_Keplerian_Elements[bytes(planet_list[i])].get_xyz(t);
            result_list[i] = string(abi.encodePacked(planet_list[i], "_", pos_x.toString(), "_", pos_y.toString(), "_", pos_z.toString() ));
        }

        return result_list;
    }

    function update_euler_matrix(string memory planet) public {
        Solar_System_Keplerian_Elements[bytes(planet)]._update_Euler_angle_transformation_matrix();
    }


    function read_euler_matrix(string memory planet) public view returns (int256[3][3] memory) {
        return Solar_System_Keplerian_Elements[bytes(planet)].Euler_angle_transformation_matrix;
    }


    function get_orbit2d_rv(string memory planet, int256 t) public view returns (int256, int256) {
        int256 res_r;
        int256 res_v;
        Orbit2D memory orbit_temp;

        orbit_temp = Solar_System_Keplerian_Elements[bytes(planet)].orbit2d;
        (res_r, res_v) = orbit2d.get_rv(t);
        return (res_r, res_v);
    }


    function test_cos(int256 angle) public pure returns (int256) {
        int256 cos_res = MathLib.cos(int(angle));
        return cos_res;
    }

    function test_sin(int256 angle) public pure returns (int256) {
        int256 sin_res = MathLib.sin(int(angle));
        return sin_res;
    }

    function test_tan(int256 angle) public pure returns (int256) {
        int256 tan_res = MathLib.tan(int(angle));
        return tan_res;
    }

    function test_arctan(int256 _ratio) public pure returns (int256) {
        int256 arctan_res = MathLib.arctan(int(_ratio));
        return arctan_res;
    }

    function test_arctan2(int256 Y, int256 X) public pure returns (int256) {
        int256 arctan2_res = MathLib.arctan2(Y, X);
        return arctan2_res;
    }

    function test_arcsin(int256 _ratio) public pure returns (int256) {
        int256 arcsin_res = MathLib.arcsin(int(_ratio));
        return arcsin_res;
    }

    function test_cart_ecl2cart_eq(int256 cart_X, int256 cart_Y, int256 cart_Z, int256 epsilon) public pure returns(int256, int256, int256) {
        int256 equat_X; 
        int256 equat_Y; 
        int256 equat_Z;
        (equat_X, equat_Y, equat_Z) = MathLib.cart_ecl2cart_eq(cart_X, cart_Y, cart_Z, epsilon);

        return(equat_X, equat_Y, equat_Z);
    }

    function test_cart2sph(int256 cart_X, int256 cart_Y, int256 cart_Z) public pure returns(int256, int256) {
        int256 alpha;
        int256 delta;
        (alpha, delta) = MathLib.cart2sph(cart_X, cart_Y, cart_Z);
        return(alpha, delta);
    }

    function test_sph_eq2sph_hor(int256 timeIndex, int256 RA, int256 DEC, int256 lat, int256 lon ) public pure returns(int256, int256) {
        int256 a;
        int256 A;
        (a, A) = MathLib.sph_eq2sph_hor(timeIndex, RA, DEC, lat, lon);
        return(a, A);
    }

    // function get_sky_positions()

    function get_sky_pos(string memory object, int256 obs_latitude, int256 obs_longitude) public view returns (int256, int256) {
        int256 pos_x;
        int256 pos_y;
        int256 pos_z;
        int256 temp_1 = 0;
        int256 temp_2 = 0;
        // int256 temp_3 = 0;
        // int256 obs_y = 0;
        // int256 obs_z = 0;

        (pos_x, pos_y, pos_z) = Solar_System_Keplerian_Elements[bytes(object)].get_xyz(0);
        // (temp_1, temp_2, temp_3) = Solar_System_Keplerian_Elements[bytes('Earth')].get_xyz(0);
        // (pos_x, pos_y, pos_z) = this.get_relative_3D_pos(object, 'Earth', 0);
        
        // pos_x = pos_x - temp_1;
        // pos_y = pos_y - temp_2;
        // pos_z = pos_z - temp_3;

        temp_1 = obs_latitude * 314159265 / (180 * 10**8);
        temp_2 = obs_longitude * 314159265 / (180 * 10**8);

        (temp_1, temp_2) = MathLib.get_sky_positions(pos_x, pos_y, pos_z, 40910000, temp_1, temp_2);

        return (temp_1, temp_2);
    }

}

