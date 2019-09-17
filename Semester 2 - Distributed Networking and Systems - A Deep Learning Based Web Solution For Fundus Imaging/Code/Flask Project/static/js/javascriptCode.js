function patient() {
    //alert('patient');
    document.getElementById('selection').innerHTML = "I am a patient"
    document.getElementById('btn_patient').style = "background-color: #FFC312"
    document.getElementById('btn_doctor').style = "background-color: white"
}

function doctor() {
    //alert('doctor');
    document.getElementById('selection').innerHTML = "I am a doctor"
    document.getElementById('btn_doctor').style = "background-color: #FFC312"
    document.getElementById('btn_patient').style = "background-color: white"
}

function submit() {
    email = document.getElementById("email").value;
    alert(email)
}