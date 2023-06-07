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

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id      trainId     category       categoryId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,       255     , "undef"      , 0            , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  0 ,       255     , "undef"      , 0            , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  0 ,       255     , "undef"      , 0            , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  0 ,       255     , "undef"      , 0            , False        , True         , (  0,  0,  0) ),
#     Label(  'background'           ,  0 ,       0       , "undef"      , 0            , False        , False        , (  0,  0,  0) ),
#     Label(  'free'                 ,  1 ,       1       , "undef"      , 0            , False        , False        , (128, 64,128) ),
#     Label(  '01'                   ,  2 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '02'                   ,  3 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '03'                   ,  4 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '04'                   ,  5 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '05'                   ,  6 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '06'                   ,  7 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '07'                   ,  8 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '08'                   ,  9 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '09'                   , 10 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '10'                   , 11 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '11'                   , 12 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '12'                   , 13 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '13'                   , 14 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '14'                   , 15 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '15'                   , 16 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '16'                   , 17 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '17'                   , 18 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '18'                   , 19 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '19'                   , 20 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '20'                   , 21 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '21'                   , 22 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '22'                   , 23 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '23'                   , 24 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '24'                   , 25 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '25'                   , 26 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '26'                   , 27 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '27'                   , 28 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '28'                   , 29 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '29'                   , 30 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '30'                   , 31 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '31'                   , 32 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '32'                   , 33 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '33'                   , 34 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '34'                   , 35 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '35'                   , 36 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '36'                   , 37 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '37'                   , 38 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '38'                   , 39 ,       0       , "undef"      , 0            , True         , False        , (  0,  0,  0) ),
#     Label(  '39'                   , 40 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '40'                   , 41 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '41'                   , 42 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#     Label(  '42'                   , 43 ,       2       , "undef"      , 0            , True         , False        , (  0,  0,142) ),
#
# ]

labels = [
    #       name                     id      trainId     category       categoryId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  255 ,     255     , "undef"      , 0            , False        , True         , (  255,  255,  255) ),
    Label(  'class'                ,  0 ,       0       , "undef"      , 0            , True         , False        , (  0,  0, 0) ),
    Label(  'anomaly'              ,  1 ,       1       , "undef"      , 0            , False        , False        , (  0,  0,  0) ),
]


def get_train_ids():
  train_ids = []
  for i in labels:
    if not i.ignoreInEval:
      train_ids.append(i.id)
  return train_ids