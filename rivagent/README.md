ANL 2025 Agent Skeleton
=======================

This skeleton contains the following folders/files:

1. *myagent.py* : A skeleton for an agent for the Automated Negotiation League (ANL).
   You should change the file name and the class name inside it to match
   your agent name (file name should be all small and class name in TitleCase)
2. *report*: A folder with latex files that you can use to write
   your 2-4 pages report. Please remember to submit a `pdf` version of the
   report.
3. *helpers*: A folder with helper files. These are not needed for your development
   1. *runner.py* : A helper module to run a tournament with your agent
5. *README.md*: This readme.

Using the Skeleton
==================

To develop your agent, the only required steps are the following:

1. [recommended] create a virtual environment, or use your favorite IDE to do
   that for you.
  - Install venv

    > python3 -m venv .venv

  - Activate the virtual environment:

    - On linux

      > source .venv/bin/activate

    - On windows (power shell)

      > call .venv\bin\activate.bat

2. [required] **Install anl**
    > pip install anl2025

3. [recommended] Change the name of the agent class from `MyAgent' to
   your-agent-name.
5. Change the implementation of whatever functions you need in the provided
   factory manager
6. [recommended] Modify the name of ``../report/myagent.tex`` to ``../report/your-agent-name.tex`` as appropriate and use it to write your
   report.
7. [recommended] You can run a simple tournament of your agent against basic
   strategies by either running ``myagent.py`` from the command line (in this folder):

    > python -m myagent.myagent
    
    or running the myagent.py file with your favorite IDE.

8. [required] **Submit your agent**: After developing your agent,
  zip ``your-agent-name`` folder into ``your-team-name_your-agent-name.zip``
  (with the pdf of the report included)  and submit it along with
  ``your-agent-name.pdf`` (after generating it from the tex file).
  This is the only file you need to submit.

*Submissions are accepted at https://scml.cs.brown.edu*

Agent Information
-----------------
Fill this section with your agent information

  - Agent Name: riv-agent
  - Team Name: natures
  - Contact Email: s250312x@st.go.tuat.ac.jp
  - Affiliation: Tokyo University of Agriculture and Technology, Department of Electrical Engineering and Computer Science
  - Country: Japan
  - Team Members:
    1. Jumpei Kawahara <s250312x@st.go.tuat.ac.jp>
