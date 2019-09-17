const url = 'process.php';
const form = document.querySelector('form');

function loadImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#original_image')
                .attr('src', e.target.result)
                .width(500)
                .height(500);
        };
        reader.readAsDataURL(input.files[0]);
    }
    document.getElementById("btn_zoom").style.visibility = "visible";
}


form.addEventListener("submit", e => {
    e.preventDefault();
    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        let file = files[i];
        formData.append('files[]', file);
        segmented = document.getElementById("segmented_image")
        segmented.src = "AI.png";
        segmented.style.width = 500 + 'px';
        segmented.style.height = 500 + 'px';
    }
    document.getElementById("annotation_form").style.visibility = "visible";
    // Here is the file is sending to the data base in the example it was done by php
    /*fetch(url, {
        method: 'POST',
        body: formData
    }).then(response => {
        console.log(response);
    });*/
});

function removeAnnotation() {
    var message, x;
    message = document.getElementById("image_annotation");
    message.value = ""
}
function sendAnnotation() {
    var message, x;
    message = document.getElementById("image_annotation");
}
/*
function editAnnotation() {
    
    var message, x;
    message = document.getElementById("image_annotation");
}*/



function imageZoom() {
    var img, img_segment, lens, result, result_segment, cx, cy, block_zoom = 0;
    img = document.getElementById("original_image");
    img_segment = document.getElementById("segmented_image")
    result = document.getElementById("zoom_image");
    result_segment = document.getElementById("zoom_image_2")
    result_segment.style.visibility = "visible";
    result.style.visibility = "visible";
    /*create lens:*/
    lens = document.createElement("DIV");
    lens.setAttribute("class", "img-zoom-lens");
    /*insert lens:*/
    img.parentElement.insertBefore(lens, img);
    /*calculate the ratio between result DIV and lens:*/
    cx = result.offsetWidth / lens.offsetWidth;
    cy = result.offsetHeight / lens.offsetHeight;
    /*set background properties for the result DIV:*/
    result.style.backgroundImage = "url('" + img.src + "')";
    result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
    result_segment.style.backgroundImage = "url('" + img_segment.src + "')";
    result_segment.style.backgroundSize = (img_segment.width * cx) + "px " + (img_segment.height * cy) + "px";
    /*execute a function when someone moves the cursor over the image, or the lens:*/
    lens.addEventListener("mousemove", moveLens);
    img.addEventListener("mousemove", moveLens);
    lens.addEventListener("mouseup", click);
    img.addEventListener("mouseup", click);
    function click() {
        /*the slider is no longer clicked:*/
        if (block_zoom == 1)
            block_zoom = 0;
        else
            block_zoom = 1;
    }
    function moveLens(e) {
        var pos, x, y;
        if (block_zoom == 0) {
            /*prevent any other actions that may occur when moving over the image:*/
            e.preventDefault();
            /*get the cursor's x and y positions:*/
            pos = getCursorPos(e);
            /*calculate the position of the lens:*/
            x = pos.x - (lens.offsetWidth / 2);
            y = pos.y - (lens.offsetHeight / 2);
            /*prevent the lens from being positioned outside the image:*/
            if (x > img.width - lens.offsetWidth) { x = img.width - lens.offsetWidth; }
            if (x < 0) { x = 0; }
            if (y > img.height - lens.offsetHeight) { y = img.height - lens.offsetHeight; }
            if (y < 0) { y = 0; }
            /*set the position of the lens:*/
            lens.style.left = x + 33 + "px";
            lens.style.top = y + 80 + "px";
            /*display what the lens "sees":*/
            result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
            result_segment.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
        }
    }
    function getCursorPos(e) {
        var a, x = 0, y = 0;
        e = e || window.event;
        /*get the x and y positions of the image:*/
        a = img.getBoundingClientRect();
        /*calculate the cursor's x and y coordinates, relative to the image:*/
        x = e.pageX - a.left;
        y = e.pageY - a.top;
        /*consider any page scrolling:*/
        x = x - window.pageXOffset;
        y = y - window.pageYOffset;
        return { x: x, y: y };
    }
}



function initComparisons() {
    var x, i;
    /*find all elements with an "overlay" class:*/
    x = document.getElementsByClassName("img-comp-overlay");
    for (i = 0; i < x.length; i++) {
        /*once for each "overlay" element:
        pass the "overlay" element as a parameter when executing the compareImages function:*/
        compareImages(x[i]);
    }
    function compareImages(img) {
        var slider, img, clicked = 0, w, h;
        /*get the width and height of the img element*/
        w = img.offsetWidth;
        h = img.offsetHeight;
        /*set the width of the img element to 50%:*/
        img.style.width = (w / 2) + "px";
        /*create slider:*/
        slider = document.createElement("DIV");
        slider.setAttribute("class", "img-comp-slider");
        /*insert slider*/
        img.parentElement.insertBefore(slider, img);
        /*position the slider in the middle:*/
        slider.style.top = (h / 2) - (slider.offsetHeight / 2) + "px";
        slider.style.left = (w / 2) - (slider.offsetWidth / 2) + "px";
        /*execute a function when the mouse button is pressed:*/
        slider.addEventListener("mousedown", slideReady);
        /*and another function when the mouse button is released:*/
        window.addEventListener("mouseup", slideFinish);
        /*or touched (for touch screens:*/
        slider.addEventListener("touchstart", slideReady);
        /*and released (for touch screens:*/
        window.addEventListener("touchstop", slideFinish);
        function slideReady(e) {
            /*prevent any other actions that may occur when moving over the image:*/
            e.preventDefault();
            /*the slider is now clicked and ready to move:*/
            clicked = 1;
            /*execute a function when the slider is moved:*/
            window.addEventListener("mousemove", slideMove);
            window.addEventListener("touchmove", slideMove);
        }
        function slideFinish() {
            /*the slider is no longer clicked:*/
            clicked = 0;
        }
        function slideMove(e) {
            var pos;
            /*if the slider is no longer clicked, exit this function:*/
            if (clicked == 0) return false;
            /*get the cursor's x position:*/
            pos = getCursorPos(e)
            /*prevent the slider from being positioned outside the image:*/
            if (pos < 0) pos = 0;
            if (pos > w) pos = w;
            /*execute a function that will resize the overlay image according to the cursor:*/
            slide(pos);
        }
        function getCursorPos(e) {
            var a, x = 0;
            e = e || window.event;
            /*get the x positions of the image:*/
            a = img.getBoundingClientRect();
            /*calculate the cursor's x coordinate, relative to the image:*/
            x = e.pageX - a.left;
            /*consider any page scrolling:*/
            x = x - window.pageXOffset;
            return x;
        }
        function slide(x) {
            /*resize the image:*/
            img.style.width = x + "px";
            /*position the slider:*/
            slider.style.left = img.offsetWidth - (slider.offsetWidth / 2) + "px";
        }
    }
}