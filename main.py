import spotipy
from keys import *
from sentiment import *

from spotipy.oauth2 import SpotifyOAuth

# spotify authorization stuff
sp_token = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = SP_CLIENT_ID,
                                               client_secret = SP_CLIENT_SECRET,
                                               redirect_uri = "http://localhost:8080", # provides an access token that can be refreshed
                                               scope = "user-read-currently-playing"))

# spotify data
raw_sp_data = sp_token.current_user_playing_track()
curr_track = raw_sp_data['item']['name']
spot_artist = raw_sp_data['item']['album']['artists'][0]['name']

print(spot_artist)

raw_lyrics = getLyrics(curr_track, spot_artist)
filtered_lyrics = filterTokens(raw_lyrics)
mostCommon(filtered_lyrics)
print(getSentiment(filtered_lyrics))
