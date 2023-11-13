window.addEventListener(`resize`, function() {
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 60
    if (imgElement.width > imgElement.height){
        imgElement.style.width = (width - padding)+ "px";
    }else{
        imgElement.style.height = (height - padding)+ "px";
    }
});

window.addEventListener('mousemove', function(e){
    if(mouse_state == 1 && select_circle != -1){
        X = e.layerX
        Y = e.layerY

        $(`#${select_circle}`).css("cx", X)
        $(`#${select_circle}`).css("cy", Y)
    }
})

window.addEventListener('mousedown', function(e){
    if(e.target.classList[0] == undefined){
        mouse_state = 1
    }
    if(e.target.classList[0] == "canvas"){
        mouse_state = 0
        // select_circle = -1
    }
})

window.addEventListener('mouseup', function(e){
    mouse_state = 0
    select_circle = -1
})