# CS5304 Final Project

#### Author: Hanyu zhang &nbsp;  Yiyuan Feng
This is the final project for cs5304. In this project, we use the data from last.fm .We examine 3 different ways in building recommendation systems. 

- The first is to use matrix factorization based on user listening frequency, 
- The second method add user’s social network to improve the rating matrix we developed in first method. 
- The last one is add tag similarity to take the user taste into consideration and change the weight of different user’s rating in rating matrix. 

We find that social network have positive affect to our prediction in some degree. 
In the future, we will present a recommendation algorithm that adapts social influence into implicit feedback.

Data discription:

This dataset contains social networking, tagging, and music artist listening information from a set of 2K users from Last.fm online music system. http://www.last.fm
   	    
   * artists.dat
   
        This file contains information about music artists listened and tagged by the users.
   
   * tags.dat
   
   	    This file contains the set of tags available in the dataset.

   * user_artists.dat
   
        This file contains the artists listened by each user.
        
        It also provides a listening count for each [user, artist] pair.

   * user_taggedartists.dat - user_taggedartists-timestamps.dat
   
        These files contain the tag assignments of artists provided by each particular user.
        
        They also contain the timestamps when the tag assignments were done.
   
   * user_friends.dat
   
        These files contain the friend relations between users in the database.
     
