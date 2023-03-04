import 'package:flutter/material.dart';
import 'package:shshacks23/SignupPage.dart';

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          SizedBox(
            height: 100,
          ),
          Center(
            child: LoginButtonSender(),
          ),
          Center(
            child: SizedBox(
              height: 50,
            ),
          ),
          SignupButtonSender()
        ],
      ),
    );
  }
}

class LoginButtonSender extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: () {},
      child: Text("Login"),
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
      child: Text("Sign Up"),
    );
  }
}
