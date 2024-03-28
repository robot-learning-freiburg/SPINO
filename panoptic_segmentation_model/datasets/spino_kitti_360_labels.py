# yapf: disable
# pylint: skip-file
#!/usr/bin/python
#
# KITTI-360 labels
#

from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels_14 = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 13 ,        8 ,         3 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        21,         4 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'traffic sign'         , 20 ,        24,         5 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         6 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         7 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,         8 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,         9 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,        10 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        11 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        12 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'two-wheeler'          , 32 ,        17,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels_14           }
# id to label object
id2label        = { label.id      : label for label in labels_14           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels_14) }
# KITTI-360 ID to cityscapes ID
kittiId2label   = { label.kittiId : label for label in labels_14           }
# category to list of label objects
category2labels = {}
for label in labels_14:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of KITTI-360 labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels_14:
        # print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
        print(" \"{:}\"".format(label.name))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
