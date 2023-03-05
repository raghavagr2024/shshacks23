import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'FirstPage.dart';
import 'GraphPage.dart';
import 'LoginPage.dart';
List<RatingsData> data = [];
var controller;
final storageRef = FirebaseStorage.instance.ref();
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const MyApp());
}
class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);


  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      home: Scaffold(
          body: FirstPage()
      ),
    );

  }
}

Future<void> getRatingsData() async {
  final FirebaseAuth auth = FirebaseAuth.instance;
  final User? user = auth.currentUser;
  var uid = "";
  if(user != null){
    uid = user.uid;
  }

  Map temp = {};
  Map ratings = {};
  await FirebaseFirestore.instance
      .collection('users')
      .doc('3V67G5v70JAMtpbGxvUS')
      .get()
      .then((DocumentSnapshot documentSnapshot) {
    if (documentSnapshot.exists) {
      temp = documentSnapshot.data() as Map;
      print("temp");
      print(temp.toString());
    }
  });
  for(int i = 0; i<temp.length;i++){
    if(temp.keys.elementAt(i) == uid){
      for(int j = 0; j<temp.values.elementAt(i).length;j++){
        ratings[j] = temp.values.elementAt(i)[j];
      }
    }
  }

  List<RatingsData> points = [];

  for(int i = 0; i<ratings.length;i++){
    points.add(RatingsData(ratings.keys.elementAt(i), ratings.values.elementAt(i)));

  }
  print('points');
  print(points.toString());

  data = points;
}
