# dragonfly-extended
ADMX-specific expansion on the project 8 dragonfly repo

## Outlook
This repo is hopefully a stop-gap, but will probably be used for at least a while.
Right now, ADMX controls are using Project 8's dragonfly repo, with this repo providing extensions on top of that to meet our needs.

There is a strong desire to migrate Project 8's dripline-python repo to the driplineorg/dripline-python repo, and to migrate most of the generic components of dragonfly into this as well.
If/when that happens, Project 8 and ADMX (and anyone else) would then likely have a repo on top of that common base with domain-specific extensions that don't belong in the generic package (providing nicer pairity between projects and placing the shared code in a neutral location).
The timeframe for actually doing this is unclear because someone would have to have available person-hours to dedicate.
