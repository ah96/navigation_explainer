def what_to_explain(control_param, locality_param):
    """
    what_to_explain does blah blah blah.

    :control_param: decodes what to explain:
        1. immediate robot action - only localvsglobal
        2. current contextualised robot action/behavior - more localvsglobal or globalvsglobal
        3. navigation history so far - both + some more info
        4. complete trajectory after reaching goal - 3 scoped on the whole start-goal navigation
    :return: wanted explanation
    """ 
    if control_param == 1:
        pass
    elif control_param == 2:
        pass
    elif control_param == 3:
        pass
    elif control_param == 4:
        pass

def when_to_explain(control_param):
    """
    when_to_explain does blah blah blah.

    :control_param: decodes what to explain:
        1. every time step
        2. when human is detected
        3. when human is need
        4. when human asks a question
    :return: wanted explanation
    """ 
    pass

def how_to_explain(control_param):
    """
    how_to_explain does blah blah blah.

    :control_param: decodes what to explain:
        1. visual
        2. textual
        3. verbal
        4. visual + textual
        5. visual + verbal
        6. textual + verbal
        7. visual+textual+verbal
    :return: wanted explanation
    """ 
    pass

def how_long_to_explain(control_param):
    """
    how_long_to_explain does blah blah blah.

    :control_param: decodes what to explain:
        1. until current action is finished
        2. until human need is fulfilled
        3. until human finishes discussion
    :return: wanted explanation
    """
    pass