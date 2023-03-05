import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'FirstPage.dart';
import 'GraphPage.dart';
import 'LoginPage.dart';
import 'package:awesome_notifications/awesome_notifications.dart';
import 'package:chat_gpt_sdk/chat_gpt_sdk.dart';
import 'package:firebase_storage/firebase_storage.dart';

List<RatingsData> data = [];
var controller;
var uid = "";
final storageRef = FirebaseStorage.instance.ref();
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();

  AwesomeNotifications().initialize(
    null,
    [
            NotificationChannel(
                    channelKey: 'basic_channel',
                    channelName: 'Basic notifications',
                    channelDescription: 'Notification channel for basic tests',
            ),
    ],
    debug: true,
  );
  Timer mytimer = Timer.periodic(Duration(seconds: 5), (timer) {
    pushNotifications();
  });

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      home: Scaffold(
          appBar: AppBar(backgroundColor: Colors.transparent, elevation: 0.0,),
          body: FirstPage()
      ),
    );

  }
}

Future<void> pushNotifications() async {
  AwesomeNotifications().isNotificationAllowed().then((isAllowed) {
    if(!isAllowed){
      AwesomeNotifications().requestPermissionToSendNotifications();
    }
  });

  var output = "";

  var prompts = [
    'Write some encouraging messages for someone who is having a bad day. Do not use the word "I".',
    'Write some encouraging messages for someone who is having a good day. Do not use the word "I".',
    'Write some encouraging messages for someone who is having a normal day. Do not use the word "I".'
  ];

  final openAI = OpenAI.instance.build(token: "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", baseOption: HttpSetup(receiveTimeout: 6000), isLogger: true);
  final request = CompleteText(prompt: prompts[0], model: kTranslateModelV3, maxTokens: 200);

  openAI.onCompleteStream(request:request).listen((response) => (output = response.toString()))
          .onError((err) {
            print("$err");
  });

  AwesomeNotifications().createNotification(
          content: NotificationContent(
            id: 10,
            channelKey: 'basic_channel',
            title: 'Hi there!',
            body: output,
          ),
        );
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
