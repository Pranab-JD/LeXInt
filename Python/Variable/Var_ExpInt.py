"""
Created on Fri Aug 13 15:31:46 2021

@author: Pranab JD

Description: -
        Contains several EXPRB and EPIRK methods.

"""

import sys
sys.path.insert(1, "../")

from EXPRB32 import EXPRB32                             #! 2nd and 3rd order
from EXPRB43 import EXPRB43                             #! 3rd and 4th order
from EXPRB53s3 import EXPRB53s3                         #! 3rd and 5th order
from EXPRB54s4 import EXPRB54s4                         #! 4th and 5th order

from EPIRK4s3 import EPIRK4s3                           #! 3rd and 4th order
from EPIRK4s3A import EPIRK4s3A                         #! 3rd and 4th order
from EPIRK5P1 import EPIRK5P1                           #! 4th and 5th order