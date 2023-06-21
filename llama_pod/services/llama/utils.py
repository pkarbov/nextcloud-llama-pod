"""
 - @copyright Copyright (c) 2023 Pavlo Karbovnyk <pkarbovn@gmail.com>
 -
 - @license AGPL-3.0-or-later
 -
 - This program is free software: you can redistribute it and/or modify
 - it under the terms of the GNU Affero General Public License as
 - published by the Free Software Foundation, either version 3 of the
 - License, or (at your option) any later version.
 -
 - This program is distributed in the hope that it will be useful,
 - but WITHOUT ANY WARRANTY; without even the implied warranty of
 - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 - GNU Affero General Public License for more details.
 -
 - You should have received a copy of the GNU Affero General Public License
 - along with this program. If not, see <http://www.gnu.org/licenses/>.
 -
 """

import os
import io
import stat
import uuid
import struct

from typing import IO
#########################################################################
# Functions for API calls
#########################################################################

"""
- @ Scan directory find all models files
-   return list of models dict
"""

def directory_scan(dir_path, ls_res):

    content = os.scandir(dir_path)

    for item in content:
        if item.is_file():

            (path, fl)  = os.path.split(item.path)

            with open(item.path, "rb") as in_file :

                try:
                    data  = lazy_load_ggml_file(in_file)
                    #print('vocab:{} embd:{} mult:{} head:{} layer:{} rot:{} file_type:{}'.format(
                    #       data[0],data[1],data[2],data[3],data[4],data[5],data[6]))

                    dc_res = {}
                    info   = item.stat()
                    dc_res['id']   = str(uuid.uuid5(uuid.NAMESPACE_DNS, fl))[:8]
                    dc_res['name'] = fl
                    dc_res['path'] = path
                    dc_res['size'] = info.st_size

                    dc_res['vocab']= data[0]
                    dc_res['embd'] = data[1]
                    dc_res['mult'] = data[2]
                    dc_res['head'] = data[3]
                    dc_res['layer']= data[4]
                    dc_res['rot']  = data[5]
                    dc_res['file_type'] = data[6]
                    dc_res['active'] = False

                    ls_res.append(dc_res)

                except Exception as ex:
                    pass

        else:
            directory_scan(item.path, ls_res)

#########################################################################
"""
- @ Read from file number of bytes
-   return byte array
"""

def must_read(fp: IO[bytes], length: int) -> bytes:
    ret = fp.read(length)
    if len(ret) < length:
        raise Exception("unexpectedly reached end of file")
    return ret

#########################################################################
"""
- @ Read from file number of bytes
-   return byte array
"""

def lazy_load_ggml_file(fp: io.BufferedReader) -> list:
    magic = must_read(fp, 4)[::-1]

    if magic in (b'ggjt',b'ggml'):
        version, = struct.unpack("i", must_read(fp, 4))
#        assert version == 1
    else:
        assert magic == b'ggml'
        version = None

    n_vocab, n_embd, n_mult, n_head, n_layer, rot, file_type = struct.unpack('<7i', must_read(fp, 28))

    return [n_vocab, n_embd, n_mult, n_head, n_layer, rot, file_type]

#########################################################################
"""
- @ Get file type descrition
-   return string
"""

def file_type_str(tp: int) -> str:

    if   tp == 0 :
        return 'ALL_F32'
    elif tp == 1 :
        return 'F16'
    elif tp == 2 :
        return 'Q4_0'
    elif tp == 3 :
        return 'Q4_1'
    elif tp == 4 :
        return 'Q4_1_SOME_F16'
    elif tp == 7 :
        return 'Q8_0'
    elif tp == 8 :
        return 'Q5_0'
    elif tp == 9 :
        return 'Q5_1'
    elif tp == 10 :
        return 'Q2_K'
    elif tp == 11 :
        return 'Q3_K_S'
    elif tp == 12 :
        return 'Q3_K_M'
    elif tp == 13 :
        return 'Q3_K_L'
    elif tp == 14 :
        return 'Q4_K_S'
    elif tp == 15 :
        return 'Q4_K_M'
    elif tp == 16 :
        return 'Q5_K_S'
    elif tp == 17 :
        return 'Q5_K_M'
    elif tp == 18 :
        return 'Q6_K'

    return "UNK"
#########################################################################
