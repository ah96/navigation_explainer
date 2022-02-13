from webcolors import rgb_to_name

#named_color = rgb_to_name((0,255,255), spec='css3')
#print(named_color)

from scipy.spatial import KDTree
from webcolors import ( CSS3_HEX_TO_NAMES, hex_to_rgb, )


def create_dict():    
    global kdt_db, names, predefined_names, rgb_values
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    #print('names: ', names)
    #print('rgb_values: ', rgb_values)

    #for i in range(0, len(names)):
    #    print(names[i] + " - " + str(rgb_values[i]) + '\n')

    #print('len(names): ', len(names)) 
    kdt_db = KDTree(rgb_values)
    

# find closest match
def convert_rgb_to_names(rgb_tuple):
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]
    #return f'closest match: {names[index]}'

# MY CODE
def create_dict_my():
    global kdt_db, names, predefined_names, rgb_values
    '''
    predefined_names = [
    'black',
    'white',
    'gray',
    'yellow',
    'red',
    'blue',
    'salmon',
    'aquamarine',
    'lightgreen',
    'violet']
    '''

    '''
    # for RAAD and IROS - reduced set of colors
    predefined_names = [
    'yellow',
    'blue',
    'white',
    'darkgray',
    'red',
    'lime'
    'black',
    ]
    '''

    # for RAAD and IROS
    predefined_names = [
    'yellow',
    'blue',
    'white',
    'darkgray',
    'red',
    'darkred',
    'maroon',
    'lime'
    'green'
    'darkgreen'
    'black',
    ]

    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        if color_name in predefined_names:
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))

    #print('names: ', names)
    #print('rgb_values: ', rgb_values)

    #for i in range(0, len(names)):
    #    print(names[i] + " - " + str(rgb_values[i]) + '\n')

    #print('len(names): ', len(names))       
    kdt_db = KDTree(rgb_values)

# find closest match
def convert_rgb_to_names_my(rgb_tuple):
    #global kdt_db
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]
    #return f'closest match: {names[index]}'

'''
# for RAAD and IROS
create_dict()
# darkgray
print(convert_rgb_to_names((180,180,180)))

# red
print(convert_rgb_to_names((255,0,0)))
# red
print(convert_rgb_to_names((220,0,0)))
# darkred
print(convert_rgb_to_names((180,0,0)))
# darkred
print(convert_rgb_to_names((150,0,0)))
# maroon
print(convert_rgb_to_names((128,0,0)))
# maroon
print(convert_rgb_to_names((90,0,0)))
# black
print(convert_rgb_to_names((50,0,0)))

# lime
print(convert_rgb_to_names((0,255,0)))
# lime
print(convert_rgb_to_names((0,220,0)))
# green
print(convert_rgb_to_names((0,180,0)))
# green
print(convert_rgb_to_names((0,150,0)))
# green
print(convert_rgb_to_names((0,128,0)))
# darkgreen
print(convert_rgb_to_names((0,90,0)))
# black
print(convert_rgb_to_names((0,50,0)))
'''

'''
create_dict()
# black
print(convert_rgb_to_names_my((0,0,0)))
# white
print(convert_rgb_to_names_my((255,255,255)))
# gray
print(convert_rgb_to_names_my((127,127,127)))
# yellow
print(convert_rgb_to_names_my((255,255,0)))
# red
print(convert_rgb_to_names_my((255,0,0)))
# blue
print(convert_rgb_to_names_my((0,0,255)))
# lime
print(convert_rgb_to_names_my((0,255,0)))
# aquamarine
print(convert_rgb_to_names_my((127,255,255)))
# salmon
print(convert_rgb_to_names_my((255,127,127)))
# lightgreen
print(convert_rgb_to_names_my((127,255,127)))
# springgreen
print(convert_rgb_to_names_my((61,255,125)))
# magenta
print(convert_rgb_to_names_my((255,0,255)))
# violet
print(convert_rgb_to_names((255,127,255)))
'''