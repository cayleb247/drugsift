const advQueryOpen = document.querySelector(".main-content p")
const advQueryDialog = document.querySelector(".main-content dialog")
const advQueryClose = document.querySelector("dialog button")

advQueryOpen.addEventListener("click", () => {
    advQueryDialog.showModal();
});

advQueryClose.addEventListener("click", () => {
    advQueryDialog.close();
})