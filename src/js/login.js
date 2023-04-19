var i = 3;
var j = 3;
while (i != 0) {
    var userName = "brain";
    var username = prompt("Enter Username");
    if (username == userName) {
        i = 4 - i;
        window.alert("Correct Username !!");
        while (j != 0) {
            var userPassword = "brain123";
            var password = prompt("Enter your password")
            if (password == userPassword) {
                j = 4 - j;
                window.alert("Correct Password !!");
                break;
            }
            else {
                j = j - 1;
                window.alert("Incorrect Password !!");
                // window.location = "/templates/index.html";
            }
        }
        break;
    }
    else {
        i = i - 1;
        window.alert("Incorrect Username !!");
        // window.location = "http://www.example.com/.";
    }
}
if (i == 0 || j == 0) {
    window.location = "https://colorlib.com/etc/404/colorlib-error-404-3/";
}
