
################ This is old Code #################
###################################################
# from streamlit.hashing import _CodeHasher

# class SessionState:

#     def __init__(self, session):
#         """Initialize SessionState instance."""
#         self.__dict__["_state"] = {
#             "data": {},
#             "hash": None,
#             "hasher": _CodeHasher(),
#             "is_rerun": False,
#             "session": session,
#         }

#     def __call__(self, **kwargs):
#         """Initialize state data once."""
#         for item, value in kwargs.items():
#             if item not in self._state["data"]:
#                 self._state["data"][item] = value

#     def __getitem__(self, item):
#         """Return a saved state value, None if item is undefined."""
#         return self._state["data"].get(item, None)
        
#     def __getattr__(self, item):
#         """Return a saved state value, None if item is undefined."""
#         return self._state["data"].get(item, None)

#     def __setitem__(self, item, value):
#         """Set state value."""
#         self._state["data"][item] = value

#     def __setattr__(self, item, value):
#         """Set state value."""
#         self._state["data"][item] = value
    
#     def clear(self):
#         """Clear session state and request a rerun."""
#         self._state["data"].clear()
#         self._state["session"].request_rerun()
    
#     def sync(self):
#         """Rerun the app with all state values up to date from the beginning to fix rollbacks."""
#         # Ensure to rerun only once to avoid infinite loops
#         # caused by a constantly changing state value at each run.
#         #
#         # Example: state.value += 1
#         if self._state["is_rerun"]:
#             self._state["is_rerun"] = False
        
#         elif self._state["hash"] is not None:
#             if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
#                 self._state["is_rerun"] = True
#                 self._state["session"].request_rerun()

#         self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

################ This is new Code #################
###################################################

"""Hack to add per-session state to Streamlit.

Usage
-----

>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'

Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'

"""
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)


    def get(**kwargs):
        """Gets a SessionState object for the current session.

        Creates a new object if necessary.

        Parameters
        ----------
        **kwargs : any
            Default values you want to add to the session state, if we're creating a
            new one.

        Example
        -------
        >>> session_state = get(user_name='', favorite_color='black')
        >>> session_state.user_name
        ''
        >>> session_state.user_name = 'Mary'
        >>> session_state.favorite_color
        'black'

        Since you set user_name above, next time your script runs this will be the
        result:
        >>> session_state = get(user_name='', favorite_color='black')
        >>> session_state.user_name
        'Mary'

        """
        # Hack to get the session object from Streamlit.

        ctx = ReportThread.get_report_ctx()

        this_session = None

        current_server = Server.get_current()
        if hasattr(current_server, '_session_infos'):
            # Streamlit < 0.56
            session_infos = Server.get_current()._session_infos.values()
        else:
            session_infos = Server.get_current()._session_info_by_id.values()

        for session_info in session_infos:
            s = session_info.session
            if (
                # Streamlit < 0.54.0
                (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
                or
                # Streamlit >= 0.54.0
                (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
                or
                # Streamlit >= 0.65.2
                (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
            ):
                this_session = s

        if this_session is None:
            raise RuntimeError(
                "Oh noes. Couldn't get your Streamlit Session object. "
                'Are you doing something fancy with threads?')

        # Got the session object! Now let's attach some state into it.

        if not hasattr(this_session, '_custom_session_state'):
            this_session._custom_session_state = SessionState(**kwargs)

        return this_session._custom_session_state
    # From https://discuss.streamlit.io/t/preserving-state-across-sidebar-pages/107
