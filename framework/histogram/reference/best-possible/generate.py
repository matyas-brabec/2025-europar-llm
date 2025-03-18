#!/usr/bin/env python3

import sys
import re
import os


def generate(template, bs, priv, ipt, vec):
    # replace placeholders with actual values

    code = re.sub(r'(\d+)\s*/\*\*BS\*\*/', str(bs), template)
    code = re.sub(r'(\d+)\s*/\*\*PRIV\*\*/', str(priv), code)
    code = re.sub(r'(\d+)\s*/\*\*IPT\*\*/', str(ipt), code)
    code = re.sub(r'(\d+)\s*/\*\*VEC\*\*/', str(vec), code)

    # make directory if it does not exist for the new file
    dirname = f"histogram-b{bs}-p{priv}-i{ipt}-v{vec}"
    os.makedirs(dirname, exist_ok=True)

    # write code to file
    with open(dirname + '/code.cu', 'w') as file:
        file.write(code)


if __name__ == '__main__':
    # load template file into a string
    with open(sys.argv[1], 'r') as file:
        template = file.read()

    for bs in [128, 256, 512, 1024]:
        for priv in [4, 8, 16, 32]:
            for ipt in [16, 32, 64, 128, 256, 512, 1024]:
                for vec in [1, 4, 8, 16]:
                    print(f"histogram-b{bs}-p{priv}-i{ipt}-v{vec}")
                    generate(template, bs, priv, ipt, vec)
