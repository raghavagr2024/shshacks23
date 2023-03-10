import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'GraphPage.dart';
import 'NostalgiaPage.dart';
import 'RecordingPage.dart';
import 'main.dart';


class HomePage extends StatefulWidget{
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _HomePage();
  }

}

class _HomePage extends State<HomePage>{
  var index = 0;
  @override
  Widget build(BuildContext context) {

    return Scaffold(
      body: (getPage(index)),
      bottomNavigationBar: BottomNavigationBar(

        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: 'Record Video',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.auto_graph),
            label: 'Graph',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.access_time),
            label: 'Nostalgia',
          ),
        ],
        currentIndex: index,

        onTap: _onItemTapped,
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.miniEndDocked,
    );
  }

  Widget getPage(int index) {
    if(index == 0){
      return CameraAwesomeApp();
    }
    else if (index == 1){
      getRatingsData();
      return GraphPage();
    }
    else {
      return VideoPlayerApp();
    }
  }
  void _onItemTapped(int i) {
    setState(() {
      index = i;
    });
  }

}

