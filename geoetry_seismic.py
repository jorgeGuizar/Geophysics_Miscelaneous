import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import geopandas as gpd
import pandas as pd
from fiona.crs import from_epsg


xmi = 575000   # Easting of bottom-left corner of grid (m)
ymi = 4710000  # Northing of bottom-left corner (m)
SL = 600       # Source line interval (m)
RL = 600       # Receiver line interval (m)
si = 100       # Source point interval (m)
ri = 100       # Receiver point interval (m)
x = 3000       # x extent of survey (m)
y = 1800       # y extent of survey (m)

#These 8 parameters allow us to construct the source and receiver pattern; the coordinates of each source and receiver location. We can calculate the number of receiver lines and source lines, as well as the number of receivers and sources for each.

#Calculate the number of receiver and source lines:

rlines = int(y/RL) + 1
slines = int(x/SL) + 1

#Calculate the number of points per line (adding 2 to straddle the edges):

rperline = int(x/ri) + 2
sperline = int(y/si) + 2

#Offset the receiver points:

shiftx = -si/2.0
shifty = -ri/2.0

#The four lists of x and y coordinates for receivers and sources are:

rcvrx, rcvry = zip(*[(xmi + rcvr*ri + shiftx, ymi + line*RL - shifty)
   for line in range(rlines)
   for rcvr in range(rperline)])

srcx, srcy = zip(*[(xmi + line*SL, ymi + src*si)
   for line in range(slines)
   for src in range(sperline)])


#plot de fuentes y receptores


plt.scatter(rcvrx, rcvry, color='red', s=10, label='Receivers')
plt.scatter(srcx,srcy, color='blue', s=10,  label='Sources')
plt.legend(loc='best')
plt.title("Source and Receiver Locations")
plt.show()


rcvrs = [Point(x, y) for x, y in zip(rcvrx, rcvry)]
srcs = [Point(x, y) for x, y in zip(srcx, srcy)]

epsg = 26911 
station_list = ['r']*len(rcvrs) + ['s']*len(srcs)
survey = gpd.GeoDataFrame({'geometry': rcvrs+srcs, 'station': station_list})
survey.crs = from_epsg(epsg)

sid = np.arange(len(survey))
survey['SID'] = sid


survey.to_file(r'C:/Users/LEGA/Documents/Geofisica/MB/survey_orig.shp')

#Midpoint calculations
#We need midpoints. There is a midpoint between every source-receiver pair.
#Hopefully it's not too inelegant to get to the midpoints now that we're using this layout object thing.


midpoint_list = [LineString([r, s]).interpolate(0.5, normalized=True)
                 for r in rcvrs
                 for s in srcs]


offsets = [r.distance(s)
           for r in rcvrs
           for s in srcs]


azimuths = [np.arctan((r.x - s.x)/(r.y - s.y))
            for r in rcvrs
            for s in srcs]


midpoints = gpd.GeoDataFrame({'geometry': midpoint_list,
                              'offset': offsets,
                              'azimuth': np.degrees(azimuths),
                              })

print(midpoints[:5])


midpoints.to_file(r'C:/Users/LEGA/Documents/Geofisica/MB/midpoints.shp')

ax = midpoints.plot()
plt.title("Midpoints")
plt.grid(True)
plt.show()

ax = midpoints.plot()
plt.title("Midpoints")
plt.grid(True)
plt.show()


midpoints['offsetx'] = offsets * np.cos(azimuths)
midpoints['offsety'] = offsets * np.sin(azimuths)
midpoints[:5].offsetx  # Easy!
#We need lists (or arrays) to pass into the matplotlib quiver plot. 
# #This takes four main parameters: x, y, u, and v, where x, y will be our coordinates, 
# #and u, v will be the offset vector for that midpoint.

#We can get at the GeoDataFrame's attributes easily, but I can't see how to get
# at the coordinates in the geometry GeoSeries (seems like a user error — it feels 
# #like it should be really easy) so I am resorting to this:


x = [m.geometry.x for i, m in midpoints.iterrows()]
y = [m.geometry.y for i, m in midpoints.iterrows()]


fig = plt.figure(figsize=(12,8))
plt.quiver(x, y, midpoints.offsetx, midpoints.offsety, units='xy', width=0.5, scale=1/0.025, pivot='mid', headlength=0)
plt.axis('equal')
plt.show()


#binning
#The bins are a new geometry, related to but separate from the survey itself, and the midpoints. We will model them as a GeoDataFrame of polygons. The steps are:

#Compute the bin centre locations with our usual list comprehension trick.
#Buffer the centres with a square.
#Gather the buffered polygons into a GeoDataFrame.

# Factor to shift the bins relative to source and receiver points
jig = si / 4.
bin_centres = gpd.GeoSeries([Point(xmi + 0.5*r*ri + jig, ymi + 0.5*s*si + jig)
                             for r in range(2*rperline - 3)
                             for s in range(2*sperline - 2)
                            ])

# Buffers are diamond shaped so we have to scale and rotate them.
scale_factor = np.sin(np.pi/4.)/2.
bin_polys = bin_centres.buffer(scale_factor*ri, 1).rotate(-45)
bins = gpd.GeoDataFrame(geometry=bin_polys)


print(bins[:3])



def bin_the_midpoints(bins, midpoints):
    b = bins.copy()
    m = midpoints.copy()
    reindexed = b.reset_index().rename(columns={'index':'bins_index'})
    joined = gpd.tools.sjoin(reindexed, m)
    bin_stats = joined.groupby('bins_index')['offset'].aggregate(fold=len, min_offset='min', max_offset='max')
    return gpd.GeoDataFrame(b.join(bin_stats))

bin_stats = bin_the_midpoints(bins, midpoints)

print(bin_stats[:10])

ax = bins.plot()
plt.show()



ax = bin_stats.plot(column="fold")
plt.title("Nominal Fold")
plt.show()

ax = bin_stats.hist(column="offset", bins=20)
plt.show()

ax = bin_stats.plot(column="min_offset")
plt.title("Minimum Offset")
plt.show()


ax = bin_stats.plot(column="max_offset")
plt.title("Maximum Offset")
plt.show()

#moving stations 
new_survey = gpd.GeoDataFrame.from_file(r'C:/Users/LEGA/Documents/Geofisica/MB/datas/adjusted.shp')
#Join the old survey to the new, based on SID.

complete = pd.merge(survey, new_survey[['geometry', 'SID']], how='outer', on='SID')
#Rename the columns.

complete.columns = ['geometry', 'station', 'SID', 'new_geometry']
print(complete[18:23])



old = gpd.GeoSeries(complete.geometry)
new = gpd.GeoSeries(complete['new_geometry'])

complete['skid'] = old.distance(new)


print(complete[15:20])

#Now that we have the new geometry, we can recompute everything, this time from the dataframe rather than from lists.

#First the midpoints
# we get a warning message in this block of code, but we can just ignore it.
import warnings
warnings.filterwarnings('ignore')

rcvrs = complete[complete['station']=='r'][complete['new_geometry'].notnull()]['new_geometry']
srcs = complete[complete['station']!='r'][complete['new_geometry'].notnull()]['new_geometry']

midpoint_list = [LineString([r, s]).interpolate(0.5, normalized=True)
                 for r in rcvrs
                 for s in srcs]
print(len(srcs), len(rcvrs), len(midpoint_list))
offsets = [r.distance(s)
           for r in rcvrs
           for s in srcs]

azimuths = [np.arctan((r.x - s.x)/(r.y - s.y))
            for r in rcvrs
            for s in srcs]


new_midpoints = gpd.GeoDataFrame({'geometry': midpoint_list,
                                  'offset': offsets,
                                  'azimuth': np.degrees(azimuths),
                                  })




new_stats = bin_the_midpoints(bins, new_midpoints).fillna(-1)


new_stats.plot(column="fold")
plt.title("Fold 3D")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.show()


new_stats.to_file('datas/new_stats.shp')












#