// Landing Page Advanced Options

const advQueryOpen = document.querySelector(".main-content p")
const advQueryDialog = document.querySelector(".main-content dialog")
const advQueryClose = document.querySelector("dialog #dialog-close")

advQueryOpen.addEventListener("click", () => {
    advQueryDialog.showModal();
});

advQueryClose.addEventListener("click", () => {
    advQueryDialog.close();
})

// Drug Profiles

const drugProfiles = document.querySelectorAll(".drug-profile")

for (drugProfile of drugProfiles) {
    addEventListener("click", () => {
        
    })
}