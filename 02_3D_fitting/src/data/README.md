## What each column means
*  `parentID`: parentID == 0 if the entry is a primary, else it would be the eventID of the primary
*  `trackID`: An Event can have multiple tracks (e.g. secondaries ) to differentiate between them you can use the trackID
*  `eventID`: Can be used to get a unique event (triggered by one primary(e.g.proton))
"Event ID, incremented for each decay. That allows in particular to know
if a coincidence is a random coincidence or not (4 bytes)"
(http://www.opengatecollaboration.org/FAQ)
*  `posX`: The position in mm of the calorimeter in x direction (depends on the simulation geometry)
*  `posY`: The position in mm of the calorimeter in y direction (depends on the simulation geometry)
*  `posZ`: The position in mm of the calorimeter in z direction (depends on the simulation geometry)
*  `PDGEncoding`: Particle Data Group - the type of the particle. The particle codes can be found at http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf. Further information could be found on page 10 in https://indico.ph.tum.de/event/3955/sessions/752/attachments/2741/3099/Day3_Physics.pdf
*  `trackLocalTime`[s]: (available starting Gate 8.0) The time that it takes a particle to complete a track
*  `time`[s]: The absolute time of a hit in the sensitive detector.
*  `stepLength`[mm]: The distance between two interactions of a particle (e.g.: distance between a gamma particle entering a sensitive volume and being scattered)
*  `trackLength`[mm]: (available starting Gate 8.0) The total distance of one particle often including multiple steps. Can also be derived by the trackLocalTime
*  `momDirX,Y,Z`: (available starting Gate 8.0) The momentum direction of a detected/absorbed particle in the sensitive detector consisting of three components that make a 3D vector
*  `processName`: The process by which the particle ended its path in the sensitive detector
*  `edep`: energy deposited (incomplete)

2020-05-26

This list has been compilated by members of the ZTT pCT-Team.
