#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import magic


def prettyprint(dct: dict):
    '''
    Utility for printing CatDogClassifier output

    Parameters
    ----------
    dct : dict of {"<FILE NAME>": <IMAGE CLASS>}

    Returns
    -------
    None.
    '''
    max_key_len = max([len(str(key)) for key in dct.keys()])
    for key in sorted(dct.keys()):
        print(key,' '*(max_key_len-len(str(key)))+':',dct[key])
        
        
def is_image(file: Union[str,bytes,bytearray]) -> bool:
    '''
    Checking if a file is an image (jpeg or png).

    Parameters
    ----------
    file : path or bytes. 

    Returns
    -------
    bool: True if the file is an image.
    '''
    if isinstance(file, str): 
        file_mimetype = magic.Magic(mime=True).from_file(file)
    elif isinstance(file, (bytes,bytearray)): 
        file_mimetype = magic.Magic(mime=True).from_buffer(file)
    else:
        return False
    return file_mimetype in ('image/jpeg','image/png')

