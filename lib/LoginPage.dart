import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shshacks23/HomePage.dart';
import 'package:video_player/video_player.dart';
import 'SignupPage.dart';
import 'main.dart';

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _auth = FirebaseAuth.instance;
  TextEditingController _email = TextEditingController();
  TextEditingController _password = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _email,
              keyboardType: TextInputType.emailAddress,
              
              decoration: InputDecoration(
                labelText: 'Email',
              ),
            ),
            SizedBox(height: 16.0),
            TextField(
              obscureText: true,
              decoration: InputDecoration(
                labelText: 'Password',
              ),
              controller: _password,
            ),
            SizedBox(height: 32.0),
            ElevatedButton(
              child: Text('Login'),
              onPressed: () async {
                print('in onpressed');
                try {
                  final user = await _auth.signInWithEmailAndPassword(
                    email: _email.text,
                    password: _password.text,
                  );
                  print('in verification');
                  try{
                    controller = VideoPlayerController.network(await storageRef.child("${uid}/${DateTime.now().year-1}-${DateTime.now().month}-${DateTime.now().day}").getDownloadURL());
                  }
                  catch(e){
                    controller =  VideoPlayerController.network("https://www.youtube.com/watch?v=ZU0f8_C5Pm0");
                  }
                  print("got controller");
                  getRatingsData();
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => HomePage()),
                  );

                } catch (e) {
                  
                  showDialog(
                    context: context,
                    builder: (BuildContext context) {
                      return AlertDialog(
                        title: Text('Login failed'),
                        content: Text(e.toString()),
                        actions: [
                          TextButton(
                            child: Text('OK'),
                            onPressed: () {
                              Navigator.of(context).pop();
                            },
                          ),
                        ],
                      );
                    },
                  );
                }
              },
            ),
            Center(
              child: SignupButtonSender()
            ),
          ],
        ),
      ),
    );
  }
}

class SignupButtonSender extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => SignupPage()),
        );
      },
      child: Text("Don't have an account? Register"),
    );
  }
}