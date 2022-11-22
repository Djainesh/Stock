from project import db
from project.com.vo.LoginVO import LoginVO
from project.com.vo.RegisterVO import RegisterVO

class LoginDAO:
    def validateLogin(self,loginVO):
        loginList = LoginVO.query.filter_by(loginUsername=loginVO.loginUsername,loginPassword=loginVO.loginPassword).all()
        return loginList

    def insertLogin(self,loginVO):
        db.session.add(loginVO)
        db.session.commit()

    def viewUser(self):
        loginList = db.session.query(RegisterVO, LoginVO)\
            .join(LoginVO, RegisterVO.register_LoginId == LoginVO.loginId)
        return loginList

    def blockUnblockUser(self, loginVO):
        db.session.merge(loginVO)
        db.session.commit()


