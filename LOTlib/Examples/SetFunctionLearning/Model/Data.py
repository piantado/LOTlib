

import re
import os
from collections import defaultdict
from LOTlib.DataAndObjects import FunctionData, Obj

CONCEPT_DIR="Concepts"
concept2data = defaultdict(list)
for pth in os.listdir(CONCEPT_DIR):

    if not re.search(r"L[34]", pth): # skip these!
        with open(CONCEPT_DIR+"/"+pth, 'r') as f:
            description = f.next() # the first line of the file

            for l in f:
                parts = re.split(r"\t", l.strip())

                # parse the true/false
                output = [ x == "#t" for x in re.findall("\#[tf]", parts[0])]

                # parse the set
                input = []
                for theobj in parts[1:]:
                    x = re.split(r",", theobj) # split within obj via commas
                    input.append( Obj(shape=x[0], color=x[1], size=int(x[2])) )

                concept2data[pth].append( FunctionData(input=input, output=output) )
