// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app"
import { getFirestore, collection, addDoc } from "firebase/firestore"
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCruZIpZxtZVoa5df0Ee4M6CJ0SKONqysc",
  authDomain: "meetcutebookrecs.firebaseapp.com",
  projectId: "meetcutebookrecs",
  storageBucket: "meetcutebookrecs.appspot.com",
  messagingSenderId: "759224877118",
  appId: "1:759224877118:web:7c4c9ec537f31593b6916f"
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// access Firestore
const db = getFirestore(app)

// reference "suggestions" colleciton
const colRef = collection(db, 'suggestions')

// add book suggestion to "suggestions" collection
const suggestionBookForm = document.querySelector('.suggestion')

suggestionBookForm.addEventListener('submit', (e) => {
    e.preventDefault()
    addDoc(colRef, {
        title: suggestionBookForm.title.value,
        author: suggestionBookForm.author.value
    })
    .then(() => {
        suggestionBookForm.reset();
    });
});