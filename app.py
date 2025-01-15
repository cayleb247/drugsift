from website import create_app
from main import main
from website.models import compoundScoringData

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
    with app.app_context():
        main("chronic thromboembolic pulmonary hypertension AND english[Language]", 5, "calebtw8@gmail.com")
        
        data = compoundScoringData.query.get(1)
        print(data)